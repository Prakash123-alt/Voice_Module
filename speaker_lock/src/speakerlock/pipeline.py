"""SpeakerLock: the streaming pipeline.

    audio in (any chunk size) -> Extractor (locked to the target) -> re-block into frames ->
    VAD -> RollingVerifier (background) -> HysteresisGate -> clean 16 kHz PCM + ChunkMeta

Output sample k corresponds to input sample k; the stream is delayed by `latency_samples`
(extractor look-back + frame re-blocking) plus compute time. Total output length == total input length
after flush().
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .buffer import RingBuffer
from .config import SAMPLE_RATE, Config
from .enroll import UserStore
from .interfaces import VAD, Embedder, Extractor, TargetCondition
from .verifier import HysteresisGate, RollingVerifier


@dataclass
class ChunkMeta:
    verified: bool               # gate open for (the end of) this output chunk
    score: Optional[float]       # latest target similarity (None until the first score)
    best_user: Optional[str]     # closed-set best match for the latest score
    latency_ms: float            # algorithmic delay + compute time of this call (excludes device buffering)
    compute_ms: float            # wall-clock time spent in process()
    vad: float                   # max speech probability in this output chunk
    start_sample: int            # stream position of the first output sample
    samples: int                 # output samples in this chunk
    failures: int                # consecutive failing verification windows
    liveness: Optional[float] = None  # heuristic replay/playback estimate for the latest score


class SpeakerLock:
    def __init__(self, cfg: Config, user_id: str, condition: TargetCondition, profiles: dict[str, np.ndarray],
                 extractor: Extractor, vad: VAD, embedder: Embedder,
                 on_unverified: Optional[Callable[[dict], None]] = None, background_verify: bool = True) -> None:
        self.cfg, self.user_id = cfg.validate(), user_id
        self.extractor, self.vad, self.embedder = extractor, vad, embedder
        self.condition, self.profiles = condition, profiles
        self._user_cb = on_unverified
        self.verifier = RollingVerifier(embedder, profiles, user_id, cfg.verifier, background=background_verify)
        self.gate = HysteresisGate(cfg.gate, user_id, on_unverified=self._on_unverified)
        self._frame = cfg.frame_samples
        self._out = RingBuffer(capacity=4 * SAMPLE_RATE)
        self._raw = RingBuffer(capacity=4 * SAMPLE_RATE)
        # second VAD on the raw input: "someone is talking" for turn boundaries, robust to suppression
        clone = getattr(vad, "clone", None)
        self._raw_vad = clone() if callable(clone) else None
        self.reset()

    # ---------------------------------------------------------------- construction
    @classmethod
    def create(cls, cfg: Optional[Config] = None, user_id: str = "default", store_dir: Optional[str] = None,
               on_unverified: Optional[Callable[[dict], None]] = None, background_verify: bool = True) -> "SpeakerLock":
        """Build all models from `cfg` and lock onto an enrolled user from the store."""
        from .embedder import build_embedder  # noqa: PLC0415
        from .extractor import build_extractor  # noqa: PLC0415
        from .vad import build_vad  # noqa: PLC0415
        cfg = (cfg or Config()).validate()
        store = UserStore(store_dir or cfg.store_dir)
        target = store.load(user_id)
        return cls(cfg, user_id, target.condition, store.profiles(),
                   build_extractor(cfg.extractor, cfg.device), build_vad(cfg.vad, cfg.frame_samples, cfg.device),
                   build_embedder(cfg.verifier, cfg.device), on_unverified, background_verify)

    @property
    def latency_samples(self) -> int:
        """Algorithmic delay: extractor look-back + worst-case frame re-blocking."""
        return self.extractor.latency_samples + self._frame

    def reset(self) -> None:
        self.extractor.start(self.condition)
        self.vad.reset()
        self.verifier.reset()
        self.gate.reset()
        self._out.clear()
        self._raw.clear()
        if self._raw_vad is not None:
            self._raw_vad.reset()
        self._pos = 0         # output samples emitted (stream position)
        self._n_in = 0

    def close(self) -> None:
        self.verifier.close()

    def _on_unverified(self, info: dict) -> None:
        if self._user_cb is not None:
            self._user_cb({"user_id": self.user_id, **info})

    # ---------------------------------------------------------------- streaming
    def _frames(self, final: bool = False) -> tuple[np.ndarray, float]:
        out, vad_max = [], 0.0
        thr = self.cfg.vad.threshold
        frames = list(self._out.frames(self._frame))
        if final and len(self._out):
            rest = self._out.pop(len(self._out))
            frames.append(np.pad(rest, (0, self._frame - rest.shape[0])))
        for frame in frames:
            p = self.vad.probability(frame)
            vad_max = max(vad_max, p)
            raw = self._raw.pop(min(frame.shape[0], len(self._raw)))
            talking = None
            if self._raw_vad is not None:
                if raw.shape[0] < frame.shape[0]:
                    raw = np.pad(raw, (0, frame.shape[0] - raw.shape[0]))
                talking = self._raw_vad.probability(raw) >= thr
            self._pos += frame.shape[0]
            score = self.verifier.push(frame, p >= thr, self._pos, talking=talking)
            if self.verifier.new_turn:
                self.gate.new_turn(self._pos)
            self.gate.update(score, self._pos, talking=p >= thr if talking is None else talking)
            if self.verifier.idle and score is None and self.gate.open != self.cfg.gate.start_open:
                self.gate.reset()     # nobody has spoken for max_age_s: the next talker starts unverified
            out.append(self.gate.apply(frame))
        y = np.concatenate(out) if out else np.zeros(0, np.float32)
        return y, vad_max

    def _meta(self, y: np.ndarray, vad_max: float, t0: float) -> ChunkMeta:
        compute = (time.perf_counter() - t0) * 1000
        s = self.gate.last_score
        return ChunkMeta(verified=self.gate.open, score=None if s is None else round(s.target, 4),
                         best_user=None if s is None else s.best_user,
                         latency_ms=round(self.latency_samples / SAMPLE_RATE * 1000 + compute, 2),
                         compute_ms=round(compute, 2), vad=round(vad_max, 3),
                         start_sample=self._pos - y.shape[0], samples=y.shape[0], failures=self.gate.failures,
                         liveness=None if s is None else s.liveness)

    def process(self, chunk: np.ndarray) -> tuple[np.ndarray, ChunkMeta]:
        """Feed any number of 16 kHz mono float32 samples; returns the clean audio that is ready."""
        t0 = time.perf_counter()
        chunk = np.asarray(chunk, np.float32).ravel()
        self._n_in += chunk.shape[0]
        self._raw.push(chunk)
        self._out.push(self.extractor.process(chunk))
        y, vad_max = self._frames()
        return y, self._meta(y, vad_max, t0)

    def flush(self) -> tuple[np.ndarray, ChunkMeta]:
        """End of stream: returns the remaining samples (total output length == total input length)."""
        t0 = time.perf_counter()
        self._out.push(self.extractor.flush())
        y, vad_max = self._frames(final=True)
        y = y[: max(0, self._n_in - (self._pos - y.shape[0]))]
        self._pos = self._n_in
        return y, self._meta(y, vad_max, t0)

    def process_file(self, audio: np.ndarray, chunk_samples: Optional[int] = None) -> tuple[np.ndarray, list[ChunkMeta]]:
        """Stream a whole array through the pipeline chunk by chunk (no lookahead); returns output + metadata."""
        n = chunk_samples or self._frame
        outs, metas = [], []
        for i in range(0, len(audio), n):
            y, m = self.process(audio[i:i + n])
            outs.append(y), metas.append(m)
        y, m = self.flush()
        outs.append(y), metas.append(m)
        return np.concatenate(outs), metas
