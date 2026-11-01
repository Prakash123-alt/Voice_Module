"""SpeakerLock as a LiveKit audio processor: clean RTC frames before they reach the agent/STT.

LiveKit `rtc.AudioFrame` carries interleaved int16 PCM (`.data`, `.sample_rate`,
`.num_channels`, `.samples_per_channel`). The adapter is intentionally duck-typed - it works
with any frame object exposing those attributes, so it does not pin a livekit version:

    from speakerlock.integrations.livekit_adapter import LiveKitSpeakerLock
    gate = LiveKitSpeakerLock(user_id="alice")
    # in your audio callback / track processor:
    frame = gate.process(frame)      # frame.data now contains only the enrolled voice
    meta  = gate.last_meta           # ChunkMeta for the last call (verified, score, liveness, ...)

Notes
-----
* Frames must be mono; multi-channel input raises (mix down upstream first).
* Sample rates other than 16 kHz need `soxr` for streaming resampling.
* `process()` is synchronous - call it from a worker/executor, not the RTC event loop.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

from ..audio_io import from_pcm16, to_pcm16
from ..config import SAMPLE_RATE, Config
from ..pipeline import ChunkMeta, SpeakerLock


class LiveKitSpeakerLock:
    def __init__(self, *, user_id: str, config: Optional[Config] = None, store_dir: Optional[str] = None,
                 on_unverified: Optional[Callable[[dict], None]] = None,
                 on_meta: Optional[Callable[[ChunkMeta], None]] = None,
                 lock: Optional[SpeakerLock] = None) -> None:
        self._lock = lock or SpeakerLock.create(config or Config(), user_id, store_dir,
                                                on_unverified=on_unverified)
        self._on_meta = on_meta
        self._to16k = self._from16k = None
        self.last_meta: Optional[ChunkMeta] = None

    def _ensure_resamplers(self, rate: int) -> None:
        if rate != SAMPLE_RATE and self._to16k is None:
            import soxr  # noqa: PLC0415
            self._to16k = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")
            self._from16k = soxr.ResampleStream(SAMPLE_RATE, rate, 1, dtype="float32")

    def process(self, frame):
        """Filter one LiveKit-style audio frame; returns a new frame of the same type/duration."""
        if getattr(frame, "num_channels", 1) != 1:
            raise ValueError("SpeakerLock expects mono frames - mix down channels upstream")
        rate = getattr(frame, "sample_rate", SAMPLE_RATE)
        self._ensure_resamplers(rate)
        raw = frame.data.tobytes() if hasattr(frame.data, "tobytes") else bytes(frame.data)
        x = from_pcm16(raw)
        if self._to16k is not None:
            x = self._to16k.resample_chunk(x)
        y, meta = self._lock.process(x)
        self.last_meta = meta
        if self._on_meta is not None:
            self._on_meta(meta)
        if self._from16k is not None:
            y = self._from16k.resample_chunk(y)
        data = to_pcm16(y)
        return type(frame)(data=data, sample_rate=rate, num_channels=1,
                           samples_per_channel=len(data) // 2)

    def close(self) -> None:
        self._lock.close()
