"""Silero VAD + a frame-level segmenter with onset debounce, hangover and pre-roll padding."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .interfaces import VADDetector
from .logging_utils import log

logger = logging.getLogger(__name__)


class SileroVAD(VADDetector):
    """Silero needs 512-sample windows at 16 kHz; we buffer 20 ms frames and
    report the most recent window probability for every frame."""

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
            win, self._buf = self._buf[: self.WINDOW], self._buf[self.WINDOW:]
            with self._torch.no_grad():
                self._last = float(self._model(self._torch.from_numpy(win), self.rate).item())
        return self._last

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._last = 0.0
        self._model.reset_states()


class EnergyVAD(VADDetector):
    """Tiny fallback used in tests / when torch is missing. Not for production."""

    def __init__(self, threshold_db: float = -40.0) -> None:
        self.threshold_db = threshold_db

    def probability(self, frame: np.ndarray) -> float:
        from .dsp import rms_db
        db = rms_db(frame)
        return float(np.clip((db - self.threshold_db) / 20 + 0.5, 0, 1))


@dataclass
class SegmentEvent:
    kind: str                      # "start" | "speech" | "end"
    audio: Optional[np.ndarray] = None
    speech_ms: int = 0             # milliseconds of speech so far (start/speech) or total (end)


@dataclass
class Segmenter:
    """Turns per-frame VAD probabilities into utterances.

    - Onset: >= min_speech_ms consecutive speech frames (debounce clicks).
    - Pre-roll: pad_ms of audio before onset is prepended so first phonemes aren't clipped.
    - Hangover: segment ends only after hangover_ms of non-speech.
    - max_segment_s: hard cut for runaway segments.
    """
    frame_ms: int = 20
    threshold: float = 0.5
    min_speech_ms: int = 120
    hangover_ms: int = 500
    pad_ms: int = 200
    max_segment_s: float = 20.0
    _pre: deque = field(init=False)
    _seg: list = field(default_factory=list, init=False)
    _onset_run: int = field(default=0, init=False)
    _silence_run: int = field(default=0, init=False)
    _speech_frames: int = field(default=0, init=False)
    _speech_start: int = field(default=0, init=False)
    active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # must hold the debounce frames *and* pad_ms before them, or the pre-roll is lost
        self._pre = deque(maxlen=max(1, (self.pad_ms + self.min_speech_ms) // self.frame_ms))

    def reset(self) -> None:
        self._pre.clear()
        self._seg.clear()
        self._onset_run = self._silence_run = self._speech_frames = self._speech_start = 0
        self.active = False

    def push(self, frame: np.ndarray, prob: float) -> Optional[SegmentEvent]:
        is_speech = prob >= self.threshold
        if not self.active:
            self._pre.append(frame)
            self._onset_run = self._onset_run + 1 if is_speech else 0
            if self._onset_run * self.frame_ms >= self.min_speech_ms:
                self.active = True
                self._seg = list(self._pre)
                self._speech_start = len(self._seg) - self._onset_run  # first speech frame, after the pad
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
        if self._silence_run * self.frame_ms >= self.hangover_ms or total_ms >= self.max_segment_s * 1000:
            audio = np.concatenate(self._seg)
            ev = SegmentEvent("end", audio=audio, speech_ms=self._speech_frames * self.frame_ms)
            self.active = False
            self._seg = []
            self._onset_run = 0
            return ev
        return SegmentEvent("speech", speech_ms=self._speech_frames * self.frame_ms)

    def current_audio(self, last_ms: Optional[int] = None) -> np.ndarray:
        if not self._seg:
            return np.zeros(0, dtype=np.float32)
        frames = self._seg if last_ms is None else self._seg[-max(1, last_ms // self.frame_ms):]
        return np.concatenate(frames)

    def current_speech_audio(self, last_ms: Optional[int] = None) -> np.ndarray:
        """Audio from speech onset — the pad_ms pre-roll excluded. Used for speaker verification,
        where pre-roll is echo/silence that would dilute the embedding of a short interruption."""
        if not self._seg:
            return np.zeros(0, dtype=np.float32)
        frames = self._seg[self._speech_start:]
        if last_ms is not None:
            frames = frames[-max(1, last_ms // self.frame_ms):]
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


def build_vad(sample_rate: int) -> VADDetector:
    try:
        return SileroVAD(sample_rate)
    except Exception as e:
        log(logger, logging.WARNING, "Silero VAD unavailable, using energy VAD (degraded)", err=str(e))
        return EnergyVAD()
