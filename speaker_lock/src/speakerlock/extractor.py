"""Target-speaker extraction (voice isolation) conditioned on the enrolled speaker.

VoxLockExtractor runs the VoxLock causal band-split-RNN TSE model (Apache-2.0, Arbaz Siddiqui,
https://github.com/arbazsiddiqui/voxlock) from its ONNX graphs. The streaming logic below follows the
reference implementation (voxlock/core.py, v1.1.1): STFT frames (512-point, hop 128, torch.stft-style
reflect centre padding) go through the `chunk` graph, which carries the LSTM state (h, c) between calls;
outputs are overlap-added with a squared-Hann normalisation. Only the LSTM state crosses calls, so the
output is identical for any chunking. Algorithmic delay: NFFT - HOP = 384 samples (24 ms).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, ExtractorConfig
from .interfaces import Extractor, TargetCondition
from .ort_utils import session

HOP, NFFT, PAD = 128, 512, 256
_WIN2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32) ** 2
_GAIN = -0.5  # the network outputs the target at about -2x (scale-invariant training); restores input level


def loudest(x: np.ndarray, seconds: float = 3.0) -> np.ndarray:
    """The loudest `seconds` of `x` (10 ms energy frames) - the reference sample VoxLock is benchmarked with."""
    x = np.asarray(x, np.float32)
    w, n = int(seconds * 100), x.shape[0] // 160
    if n <= w:
        return x
    e = np.square(x[: n * 160].reshape(n, 160)).mean(1)
    st = int(np.convolve(e, np.ones(w), "valid").argmax()) * 160
    return x[st: st + int(seconds * SAMPLE_RATE)]


class VoxLockExtractor(Extractor):
    latency_samples = NFFT - HOP

    def __init__(self, precision: str = "int8-full", threads: int = 2, model_dir: Optional[str] = None,
                 device: str = "cpu") -> None:
        from .models import VOXLOCK_PRECISIONS, voxlock  # noqa: PLC0415
        if model_dir:
            from pathlib import Path  # noqa: PLC0415
            enroll_name, chunk_name = VOXLOCK_PRECISIONS[precision]
            enroll_p, chunk_p = Path(model_dir) / f"{enroll_name}.onnx", Path(model_dir) / f"{chunk_name}.onnx"
        else:
            enroll_p, chunk_p = voxlock(precision)
        self._enroll = session(enroll_p, threads=threads, device=device)
        self._chunk = session(chunk_p, threads=threads, device=device)
        self._cond: Optional[TargetCondition] = None

    # ---------------------------------------------------------------- enrollment
    def condition(self, sample: np.ndarray) -> TargetCondition:
        sample = loudest(np.asarray(sample, np.float32), 3.0)
        if sample.shape[0] < SAMPLE_RATE:
            raise ValueError("reference sample must contain at least 1 s of audio (3 s+ recommended)")
        mag, ctx = self._enroll.run(None, {"enroll": sample[None]})
        return TargetCondition({"enroll_mag": mag, "context": ctx})

    # ---------------------------------------------------------------- streaming
    def start(self, cond: TargetCondition) -> None:
        self._cond = cond
        self._h = np.zeros((6, 32, 256), np.float32)
        self._c = np.zeros_like(self._h)
        self._head = np.zeros(0, np.float32)   # first samples, until reflect padding can be built
        self._buf: Optional[np.ndarray] = None  # padded input not yet consumed by a frame
        self._ola = np.zeros(NFFT, np.float32)
        self._env = np.zeros(NFFT, np.float32)
        self._drop = 0                          # leading padded samples still to drop from the output
        self._n_in = self._n_out = 0            # so flush() emits exactly as many samples as came in

    def _emit(self, y: np.ndarray) -> np.ndarray:
        y = y[: max(0, self._n_in - self._n_out)]
        self._n_out += y.shape[0]
        return y

    def _run(self) -> np.ndarray:
        assert self._cond is not None and self._buf is not None
        if self._buf.shape[0] < NFFT:
            return np.zeros(0, np.float32)
        k = 1 + (self._buf.shape[0] - NFFT) // HOP           # all complete frames in one ONNX call
        seg = self._buf[: (k - 1) * HOP + NFFT]
        frames, self._h, self._c = self._chunk.run(None, {"wav": seg[None], **self._cond.arrays,
                                                          "h": self._h, "c": self._c})
        ola = np.concatenate([self._ola, np.zeros(k * HOP, np.float32)])
        env = np.concatenate([self._env, np.zeros(k * HOP, np.float32)])
        for j in range(k):
            ola[j * HOP: j * HOP + NFFT] += frames[0, j]
            env[j * HOP: j * HOP + NFFT] += _WIN2
        done = k * HOP                                        # samples no later frame will touch
        y = ola[:done] / np.where(env[:done] > 1e-11, env[:done], 1.0)
        self._ola, self._env, self._buf = ola[done:], env[done:], self._buf[done:]
        cut = min(self._drop, y.shape[0])
        self._drop -= cut
        return (y[cut:] * _GAIN).astype(np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self._cond is None:
            raise RuntimeError("call start(condition) before process()")
        chunk = np.asarray(chunk, np.float32)
        self._n_in += chunk.shape[0]
        if self._buf is None:
            self._head = np.concatenate([self._head, chunk])
            if self._head.shape[0] <= PAD:
                return np.zeros(0, np.float32)
            self._buf = np.concatenate([self._head[1: PAD + 1][::-1], self._head])
            self._drop = PAD
        else:
            self._buf = np.concatenate([self._buf, chunk])
        return self._emit(self._run())

    def flush(self) -> np.ndarray:
        if self._buf is None:   # fewer than PAD samples ever arrived: nothing meaningful to isolate
            y = np.zeros(self._head.shape[0], np.float32)
        else:
            x = np.concatenate([self._buf, self._buf[-PAD - 1: -1][::-1]])
            nfr = 1 + max(0, x.shape[0] - NFFT) // HOP
            self._buf = np.pad(x, (0, max(0, (nfr - 1) * HOP + NFFT - x.shape[0])))
            y = self._run()
            tail = self._ola / np.where(self._env > 1e-11, self._env, 1.0) * _GAIN
            y = np.concatenate([y, tail.astype(np.float32)])
        y = self._emit(y)
        if y.shape[0] < self._n_in - self._n_out:   # pad (only possible for ultra-short streams)
            y = np.concatenate([y, np.zeros(self._n_in - self._n_out - y.shape[0], np.float32)])
            self._n_out = self._n_in
        self._buf, self._head = None, np.zeros(0, np.float32)
        return y


class PassthroughExtractor(Extractor):
    """No isolation (for A/B tests and machines too slow for the TSE model)."""

    latency_samples = 0

    def condition(self, sample: np.ndarray) -> TargetCondition:
        return TargetCondition({})

    def start(self, cond: TargetCondition) -> None:
        pass

    def process(self, chunk: np.ndarray) -> np.ndarray:
        return np.asarray(chunk, np.float32).copy()

    def flush(self) -> np.ndarray:
        return np.zeros(0, np.float32)


def build_extractor(cfg: ExtractorConfig, device: str = "cpu") -> Extractor:
    if cfg.backend == "voxlock":
        return VoxLockExtractor(cfg.precision, cfg.threads, cfg.model_dir, device)
    return PassthroughExtractor()
