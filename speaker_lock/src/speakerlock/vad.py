"""Voice activity detection: Silero (ONNX, no torch) and an energy fallback."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, VADConfig
from .interfaces import VAD
from .ort_utils import session


class SileroVAD(VAD):
    """Silero VAD v5/v6 ONNX. Native window is 512 samples (32 ms) with a 64-sample context.
    Other frame sizes are re-blocked internally; the returned value is the max probability of the
    512-sample windows completed in that frame (or the last one if none completed)."""

    WINDOW, CONTEXT = 512, 64

    def __init__(self, frame_samples: int = 512, model_path: Optional[str] = None, device: str = "cpu") -> None:
        from .models import silero_vad  # noqa: PLC0415
        self.frame_samples = frame_samples
        self._sess = session(model_path or silero_vad(), threads=1, device=device)
        self.reset()

    def clone(self) -> "SileroVAD":
        """Independent stream state, shared ONNX session (cheap)."""
        c = object.__new__(SileroVAD)
        c.frame_samples, c._sess = self.frame_samples, self._sess
        c.reset()
        return c

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros((1, self.CONTEXT), np.float32)
        self._pending = np.zeros(0, np.float32)
        self._last = 0.0

    def _window(self, w: np.ndarray) -> float:
        x = np.concatenate([self._context, w[None]], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state,
                                                 "sr": np.array(SAMPLE_RATE, np.int64)})
        self._context = x[:, -self.CONTEXT:]
        return float(out.ravel()[0])

    def probability(self, frame: np.ndarray) -> float:
        self._pending = np.concatenate([self._pending, np.asarray(frame, np.float32)])
        probs = []
        while self._pending.shape[0] >= self.WINDOW:
            probs.append(self._window(self._pending[:self.WINDOW]))
            self._pending = self._pending[self.WINDOW:]
        if probs:
            self._last = max(probs)
        return self._last


class EnergyVAD(VAD):
    """Degraded fallback: speech = frame energy >= adaptive noise floor (min of last 2 s) + margin dB."""

    def __init__(self, frame_samples: int = 512, margin_db: float = 12.0) -> None:
        self.frame_samples, self.margin = frame_samples, margin_db
        self.reset()

    def reset(self) -> None:
        self._floor: deque[float] = deque(maxlen=max(1, 2 * SAMPLE_RATE // self.frame_samples))

    def clone(self) -> "EnergyVAD":
        return EnergyVAD(self.frame_samples, self.margin)

    def probability(self, frame: np.ndarray) -> float:
        db = 10 * np.log10(float(np.mean(np.square(frame))) + 1e-10)
        if db > -90:
            self._floor.append(db)
        floor = min(self._floor) if self._floor else -90.0
        return float(np.clip((db - floor - self.margin) / 6 + 0.5, 0, 1))


def build_vad(cfg: VADConfig, frame_samples: int, device: str = "cpu") -> VAD:
    if cfg.backend == "silero":
        return SileroVAD(frame_samples, cfg.model_path, device)
    return EnergyVAD(frame_samples)
