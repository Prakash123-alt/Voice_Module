"""Acoustic Echo Cancellation — removes PC speaker output from the mic.

Adapted from VoiceModule voice/aec.py (same author).
Key difference: the reference signal comes from the WASAPI loopback (all PC
audio), not just the assistant's TTS output.

Backends
--------
WebRTCAEC  — WebRTC AEC3 via livekit APM (Apache-2.0).  Default.
GateAEC    — Mutes the mic while the reference carries signal (no barge-in).
NoAEC      — Pass-through; use with headphones.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from voice_firewall._logging import log
from voice_firewall.audio.dsp import to_float32, to_int16

logger = logging.getLogger(__name__)


class AECProcessor(ABC):
    available: bool = True

    @abstractmethod
    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        """Return mic with the reference echo removed."""

    def reset(self) -> None:
        pass


class WebRTCAEC(AECProcessor):
    """WebRTC AEC3 via livekit AudioProcessingModule (prebuilt wheels, Apache-2.0)."""
    available = True

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 320,
        delay_ms: int | str = "auto",
        noise_suppression: bool = False,
        high_pass: bool = True,
    ) -> None:
        from livekit import rtc
        self._rtc = rtc
        self.rate = sample_rate
        self.frame = frame_samples
        self._sub = sample_rate // 100  # 10 ms sub-frames
        if frame_samples % self._sub:
            raise ValueError("frame_samples must be a multiple of 10 ms for WebRTC APM")
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=high_pass,
            auto_gain_control=False,
        )
        self._delay_hint = 0 if delay_ms == "auto" else int(delay_ms)
        log(logger, logging.INFO, "AEC ready", backend="webrtc-aec3", delay_hint_ms=delay_ms)

    def _frame(self, x: np.ndarray) -> object:
        from livekit import rtc
        return rtc.AudioFrame(
            data=to_int16(x).tobytes(),
            sample_rate=self.rate,
            num_channels=1,
            samples_per_channel=x.shape[0],
        )

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        ref = reference if reference is not None else np.zeros_like(mic)
        out = np.empty_like(mic)
        for i in range(0, mic.shape[0], self._sub):
            r = self._frame(ref[i : i + self._sub])
            self._apm.process_reverse_stream(r)
            self._apm.set_stream_delay_ms(self._delay_hint)
            m = self._frame(mic[i : i + self._sub])
            self._apm.process_stream(m)
            out[i : i + self._sub] = to_float32(np.frombuffer(m.data, dtype=np.int16))
        return out


class GateAEC(AECProcessor):
    """Fallback: mutes the mic while the reference carries signal."""
    available = False  # barge-in disabled when this is active

    def __init__(self, frame_samples: int, tail_frames: int = 10) -> None:
        self.frame_samples = frame_samples
        self.tail_frames = tail_frames
        self._tail = 0
        log(logger, logging.WARNING,
            "AEC unavailable — gating mic while speaker active (barge-in disabled)")

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        if reference is not None and np.max(np.abs(reference)) > 1e-4:
            self._tail = self.tail_frames
        if self._tail > 0:
            self._tail -= 1
            return np.zeros_like(mic)
        return mic

    def reset(self) -> None:
        self._tail = 0


class NoAEC(AECProcessor):
    """Pass-through; use with headphones (no echo path)."""
    available = True

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        return mic


def build_aec(
    backend: str,
    sample_rate: int,
    frame_samples: int,
    delay_ms: int | str = "auto",
) -> AECProcessor:
    if backend == "none":
        log(logger, logging.INFO, "AEC disabled (headphones assumed)")
        return NoAEC()
    if backend == "gate":
        return GateAEC(frame_samples)
    if backend == "webrtc":
        try:
            return WebRTCAEC(sample_rate, frame_samples, delay_ms)
        except Exception as e:
            log(logger, logging.WARNING, "WebRTC AEC unavailable — falling back to gate", err=str(e))
            return GateAEC(frame_samples)
    log(logger, logging.WARNING, "unknown AEC backend — using gate", backend=backend)
    return GateAEC(frame_samples)
