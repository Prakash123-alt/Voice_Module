"""Silero VAD + frame-level segmenter.

Adapted from VoiceModule voice/vad.py (same author).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from voice_firewall._logging import log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAD detectors
# ---------------------------------------------------------------------------

class SileroVAD:
    """Silero VAD — buffers 20 ms frames into 512-sample windows."""
    WINDOW = 512

    def __init__(self, sample_rate: int = 16000) -> None:
        import torch
        from silero_vad import load_silero_vad
        torch.set_num_threads(1)
        self._torch = torch
        self._model = load_silero_vad()
        self.rate = sample_rate
        self._buf = np.zeros(0, dtype=np.float32)
        self._last = 0.0

    def probability(self, frame: np.ndarray) -> float:
        self._buf = np.concatenate([self._buf, frame])
        while self._buf.shape[0] >= self.WINDOW:
            win, self._buf = self._buf[: self.WINDOW], self._buf[self.WINDOW :]
            with self._torch.no_grad():
                self._last = float(
                    self._model(self._torch.from_numpy(win), self.rate).item()
                )
        return self._last

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._last = 0.0
        self._model.reset_states()


class EnergyVAD:
    """Energy-based fallback VAD (no model; for tests / when torch is missing)."""

    def __init__(self, threshold_db: float = -40.0) -> None:
        self.threshold_db = threshold_db

    def probability(self, frame: np.ndarray) -> float:
        from voice_firewall.audio.dsp import rms_db
        db = rms_db(frame)
        return float(np.clip((db - self.threshold_db) / 20 + 0.5, 0, 1))

    def reset(self) -> None:
        pass


def build_vad(sample_rate: int) -> SileroVAD | EnergyVAD:
    try:
        return SileroVAD(sample_rate)
    except Exception as e:
        log(logger, logging.WARNING,
            "Silero VAD unavailable — using energy VAD (degraded)", err=str(e))
        return EnergyVAD()


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------

@dataclass
class SegmentEvent:
    kind: str                        # "start" | "speech" | "end"
    audio: Optional[np.ndarray] = None
    speech_ms: int = 0


@dataclass
class Segmenter:
    """Accumulates VAD probabilities into utterances.

    - Onset debounce: speech must exceed threshold for >= min_speech_ms.
    - Pre-roll: pad_ms of audio before onset is prepended.
    - Hangover: segment ends only after hangover_ms of silence.
    - Hard cut at max_segment_s.
    """
    frame_ms: int = 20
    threshold: float = 0.5
    min_speech_ms: int = 120
    hangover_ms: int = 500
    pad_ms: int = 200
    max_segment_s: float = 20.0

    _pre: object = field(init=False)        # deque — set in __post_init__
    _seg: list[np.ndarray] = field(default_factory=list, init=False)
    _onset_run: int = field(default=0, init=False)
    _silence_run: int = field(default=0, init=False)
    _speech_frames: int = field(default=0, init=False)
    _speech_start: int = field(default=0, init=False)
    active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        from collections import deque
        self._pre: object = deque(
            maxlen=max(1, (self.pad_ms + self.min_speech_ms) // self.frame_ms)
        )

    def reset(self) -> None:
        from collections import deque
        self._pre = deque(
            maxlen=max(1, (self.pad_ms + self.min_speech_ms) // self.frame_ms)
        )
        self._seg = []
        self._onset_run = self._silence_run = self._speech_frames = self._speech_start = 0
        self.active = False

    def push(self, frame: np.ndarray, prob: float) -> Optional[SegmentEvent]:
        from collections import deque
        pre: deque[np.ndarray] = self._pre  # type: ignore[assignment]
        is_speech = prob >= self.threshold
        if not self.active:
            pre.append(frame)
            self._onset_run = self._onset_run + 1 if is_speech else 0
            if self._onset_run * self.frame_ms >= self.min_speech_ms:
                self.active = True
                self._seg = list(pre)
                self._speech_start = len(self._seg) - self._onset_run
                self._speech_frames, self._silence_run = self._onset_run, 0
                return SegmentEvent("start", speech_ms=self._speech_frames * self.frame_ms)
            return None
        self._seg.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
        total_ms = len(self._seg) * self.frame_ms
        if (self._silence_run * self.frame_ms >= self.hangover_ms
                or total_ms >= self.max_segment_s * 1000):
            audio = np.concatenate(self._seg)
            ev = SegmentEvent("end", audio=audio,
                              speech_ms=self._speech_frames * self.frame_ms)
            self.active = False
            self._seg = []
            self._onset_run = 0
            return ev
        return SegmentEvent("speech", speech_ms=self._speech_frames * self.frame_ms)

    def current_speech_audio(self, last_ms: Optional[int] = None) -> np.ndarray:
        """Audio from speech onset (pre-roll excluded) — for speaker embedding."""
        if not self._seg:
            return np.zeros(0, dtype=np.float32)
        frames = self._seg[self._speech_start :]
        if last_ms is not None:
            frames = frames[-max(1, last_ms // self.frame_ms):]
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
