"""Acoustic echo cancellation using the TTS output as far-end reference.

WebRTCAEC: WebRTC AEC3 (livekit APM bindings). Default; prebuilt wheels on Windows/macOS/Linux.

SpeexAEC: speexdsp echo canceller fed with the reference delayed by `delay_ms`
(playback->capture latency). If delay is "auto", a cross-correlation estimator
accumulates play/capture audio during the first playback and locks the delay in.

GateAEC: fallback that zeroes the mic while the speaker is active (no barge-in).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from .dsp import estimate_delay_samples, to_float32, to_int16
from .interfaces import AECProcessor
from .logging_utils import log

logger = logging.getLogger(__name__)


class DelayLine:
    """Delays the reference stream by N frames so it lines up with the mic capture."""

    def __init__(self, frame_samples: int, delay_frames: int, max_frames: int) -> None:
        self.frame_samples = frame_samples
        self._buf: deque = deque(maxlen=max(max_frames, delay_frames) + 1)
        self.delay_frames = delay_frames

    def push(self, ref: Optional[np.ndarray]) -> np.ndarray:
        self._buf.append(ref if ref is not None else np.zeros(self.frame_samples, dtype=np.float32))
        idx = len(self._buf) - 1 - self.delay_frames
        return self._buf[idx] if idx >= 0 else np.zeros(self.frame_samples, dtype=np.float32)


class DelayEstimator:
    def __init__(self, sample_rate: int, max_delay_ms: int, min_seconds: float) -> None:
        self.max_lag = int(sample_rate * max_delay_ms / 1000)
        self.need = int(sample_rate * min_seconds)
        self.rate = sample_rate
        self._ref: list[np.ndarray] = []
        self._mic: list[np.ndarray] = []
        self._n = 0
        self._silent_run = 0
        self.result_ms: Optional[int] = None

    def _clear(self) -> None:
        self._ref.clear()
        self._mic.clear()
        self._n = 0

    def push(self, mic: np.ndarray, ref: np.ndarray) -> Optional[int]:
        if self.result_ms is not None:
            return self.result_ms
        silent = np.max(np.abs(ref)) < 1e-4
        self._silent_run = self._silent_run + 1 if silent else 0
        if silent and (not self._ref or self._silent_run * mic.shape[0] > self.max_lag):
            self._clear()  # playback ended before we had enough; start fresh on the next one
            return None
        self._ref.append(ref)
        self._mic.append(mic)
        self._n += mic.shape[0]
        if self._n < self.need + self.max_lag:
            return None
        lag, conf = estimate_delay_samples(np.concatenate(self._ref), np.concatenate(self._mic), self.max_lag)
        self._clear()
        if conf < 4.0:  # not a clear peak - keep trying on the next playback
            log(logger, logging.DEBUG, "aec delay estimate inconclusive", confidence=round(conf, 2))
            return None
        self.result_ms = int(round(lag * 1000 / self.rate))
        log(logger, logging.INFO, "aec delay estimated", delay_ms=self.result_ms, confidence=round(conf, 1))
        return self.result_ms


class SpeexAEC(AECProcessor):
    available = True

    def __init__(self, sample_rate: int, frame_samples: int, filter_length_ms: int = 200,
                 delay_ms: int | str = "auto", max_delay_ms: int = 400, auto_estimate_min_s: float = 1.0) -> None:
        from speexdsp import EchoCanceller
        self.rate, self.frame = sample_rate, frame_samples
        self._filter_len = int(sample_rate * filter_length_ms / 1000)
        self._ec = EchoCanceller.create(frame_samples, self._filter_len, sample_rate)
        frame_ms = frame_samples * 1000 // sample_rate
        self._frame_ms = frame_ms
        self._max_frames = max_delay_ms // frame_ms
        self._estimator = DelayEstimator(sample_rate, max_delay_ms, auto_estimate_min_s) if delay_ms == "auto" else None
        initial = 0 if delay_ms == "auto" else int(delay_ms) // frame_ms
        self._delay = DelayLine(frame_samples, initial, self._max_frames)
        log(logger, logging.INFO, "AEC ready", backend="speex", delay_ms=delay_ms, filter_ms=filter_length_ms)

    @property
    def delay_ms(self) -> int:
        return self._delay.delay_frames * self._frame_ms

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        ref = self._delay.push(reference)
        if self._estimator is not None and self._estimator.result_ms is None and reference is not None:
            est = self._estimator.push(mic, reference)
            if est is not None:
                self._delay.delay_frames = min(self._max_frames, max(0, est // self._frame_ms))
                self._ec = type(self._ec).create(self.frame, self._filter_len, self.rate)  # re-adapt with new delay
        out = self._ec.process(to_int16(mic).tobytes(), to_int16(ref).tobytes())
        return to_float32(np.frombuffer(out, dtype=np.int16))

    def reset(self) -> None:
        pass  # speex adapts continuously; resetting mid-stream hurts more than it helps


class WebRTCAEC(AECProcessor):
    """WebRTC AEC3 via livekit's on-device AudioProcessingModule (prebuilt wheels incl. Windows).
    No network: only the local APM FFI is used. APM requires 10 ms frames, so each 20 ms frame
    is processed as two halves. AEC3 has its own internal delay estimator; `delay_ms` (if an int)
    is passed as the stream-delay hint, "auto" leaves it to AEC3."""
    available = True

    def __init__(self, sample_rate: int, frame_samples: int, delay_ms: int | str = "auto",
                 noise_suppression: bool = False, high_pass: bool = True) -> None:
        from livekit import rtc
        self._rtc = rtc
        self.rate, self.frame = sample_rate, frame_samples
        self._sub = sample_rate // 100  # 10 ms
        if frame_samples % self._sub:
            raise ValueError("frame size must be a multiple of 10 ms for WebRTC APM")
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=noise_suppression,
                                              high_pass_filter=high_pass, auto_gain_control=False)
        self._delay_hint = 0 if delay_ms == "auto" else int(delay_ms)
        log(logger, logging.INFO, "AEC ready", backend="webrtc-aec3", delay_hint_ms=delay_ms)

    def _frame(self, x: np.ndarray):
        return self._rtc.AudioFrame(data=to_int16(x).tobytes(), sample_rate=self.rate, num_channels=1,
                                    samples_per_channel=x.shape[0])

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        ref = reference if reference is not None else np.zeros_like(mic)
        out = np.empty_like(mic)
        for i in range(0, mic.shape[0], self._sub):
            r = self._frame(ref[i:i + self._sub])
            self._apm.process_reverse_stream(r)
            self._apm.set_stream_delay_ms(self._delay_hint)
            m = self._frame(mic[i:i + self._sub])
            self._apm.process_stream(m)
            out[i:i + self._sub] = to_float32(np.frombuffer(m.data, dtype=np.int16))
        return out


class GateAEC(AECProcessor):
    """Fallback: mute the mic whenever the reference carries signal (plus a short tail)."""
    available = False

    def __init__(self, frame_samples: int, tail_frames: int = 10) -> None:
        self.frame_samples, self.tail_frames, self._tail = frame_samples, tail_frames, 0
        log(logger, logging.WARNING, "AEC unavailable - gating mic while speaking (barge-in disabled)")

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        if reference is not None and np.max(np.abs(reference)) > 1e-4:
            self._tail = self.tail_frames
        if self._tail > 0:
            self._tail -= 1
            return np.zeros_like(mic)
        return mic


class NoAEC(AECProcessor):
    available = True  # user asserts headphones: no echo path, barge-in allowed

    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        return mic


def build_aec(cfg, sample_rate: int, frame_samples: int) -> AECProcessor:
    if not cfg.enabled or cfg.backend == "none":
        log(logger, logging.INFO, "AEC disabled (assuming headphones)")
        return NoAEC()
    builders = {
        "webrtc": lambda: WebRTCAEC(sample_rate, frame_samples, cfg.delay_ms),
        "speex": lambda: SpeexAEC(sample_rate, frame_samples, cfg.filter_length_ms, cfg.delay_ms, cfg.max_delay_ms,
                                  cfg.auto_estimate_min_s),
    }
    # try the requested backend first, then the other real AEC, before degrading to gating
    order = [cfg.backend] + [b for b in builders if b != cfg.backend] if cfg.backend in builders else []
    for name in order:
        try:
            return builders[name]()
        except Exception as e:
            log(logger, logging.WARNING, "AEC backend not available", backend=name, err=str(e))
    if cfg.gate_when_speaking_if_unavailable or cfg.backend == "gate":
        return GateAEC(frame_samples)
    log(logger, logging.WARNING, "no AEC and gating disabled - the assistant may hear itself")
    return NoAEC()
