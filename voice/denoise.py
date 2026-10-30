"""Neural noise suppression. RNNoise (streaming, 10 ms @ 48 kHz) is the default;
DeepFilterNet is offered for higher quality at higher CPU cost; `none` passes through."""
from __future__ import annotations

import logging

import numpy as np

from .dsp import FrameBuffer, resample, to_float32, to_int16
from .interfaces import Denoiser
from .logging_utils import log

logger = logging.getLogger(__name__)


class PassthroughDenoiser(Denoiser):
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


class RNNoiseDenoiser(Denoiser):
    """Wraps pyrnnoise. Resamples 16 kHz -> 48 kHz, denoises 480-sample frames, back to 16 kHz.
    One 20 ms input frame maps to exactly two RNNoise frames, so output stays frame-aligned."""

    def __init__(self, sample_rate: int = 16000) -> None:
        # Low-level ctypes API: the high-level RNNoise class drags in an ffmpeg resampling graph.
        from pyrnnoise.rnnoise import FRAME_SIZE, create, destroy, process_mono_frame
        self._create, self._destroy, self._process = create, destroy, process_mono_frame
        self._frame = FRAME_SIZE  # 480 @ 48 kHz
        self.rate = sample_rate
        self._state = create()
        self._fb = FrameBuffer(FRAME_SIZE)
        self.last_speech_prob = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        hi = resample(frame, self.rate, 48000)
        out = []
        for f in self._fb.push(hi):
            den, prob = self._process(self._state, to_int16(f))
            self.last_speech_prob = float(prob)
            out.append(to_float32(den))
        if not out:
            return np.zeros_like(frame)
        y = resample(np.concatenate(out), 48000, self.rate)
        n = frame.shape[0]
        return y[:n] if y.shape[0] >= n else np.pad(y, (0, n - y.shape[0]))  # guard against rounding drift

    def reset(self) -> None:
        self._destroy(self._state)
        self._state = self._create()
        self._fb = FrameBuffer(self._frame)

    def __del__(self) -> None:
        state = getattr(self, "_state", None)
        if state is not None:
            self._destroy(state)


class DeepFilterNetDenoiser(Denoiser):
    """DeepFilterNet operates best on larger blocks; we process a sliding 100 ms window
    and return the newest 20 ms, which adds ~0 algorithmic latency but more CPU."""

    def __init__(self, sample_rate: int = 16000, block_frames: int = 5) -> None:
        import torch
        from df.enhance import enhance, init_df
        self.rate = sample_rate
        self._model, self._state, _ = init_df(log_level="ERROR")
        self._enhance, self._torch = enhance, torch
        self._block_frames = block_frames
        self._history: list[np.ndarray] = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        self._history.append(frame)
        self._history = self._history[-self._block_frames:]
        block = resample(np.concatenate(self._history), self.rate, self._state.sr())
        with self._torch.no_grad():
            y = self._enhance(self._model, self._state, self._torch.from_numpy(block)[None, :]).squeeze(0).numpy()
        y = resample(y, self._state.sr(), self.rate)
        return y[-frame.shape[0]:].astype(np.float32)

    def reset(self) -> None:
        self._history.clear()


def build_denoiser(backend: str, sample_rate: int) -> Denoiser:
    try:
        if backend == "rnnoise":
            return RNNoiseDenoiser(sample_rate)
        if backend == "deepfilternet":
            return DeepFilterNetDenoiser(sample_rate)
    except Exception as e:
        log(logger, logging.WARNING, "denoiser unavailable, passing audio through", backend=backend, err=str(e))
    return PassthroughDenoiser()
