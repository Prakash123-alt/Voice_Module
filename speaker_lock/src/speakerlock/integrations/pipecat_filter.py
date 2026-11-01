"""SpeakerLock as a Pipecat audio input filter (runs before Pipecat's VAD and STT).

    from pipecat.transports.base_transport import TransportParams
    from speakerlock.integrations.pipecat_filter import SpeakerLockFilter

    params = TransportParams(audio_in_enabled=True, audio_in_sample_rate=16000,
                             audio_in_filter=SpeakerLockFilter(user_id="alice"))
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

from ..audio_io import from_pcm16, to_pcm16
from ..config import SAMPLE_RATE, Config
from ..pipeline import ChunkMeta, SpeakerLock


class SpeakerLockFilter(BaseAudioFilter):
    """Keeps only the enrolled user's voice. Input/output: 16-bit mono PCM at the transport rate
    (16 kHz natively; other rates need `soxr` for streaming resampling)."""

    def __init__(self, *, user_id: str, config: Optional[Config] = None, store_dir: Optional[str] = None,
                 on_unverified: Optional[Callable[[dict], None]] = None,
                 on_meta: Optional[Callable[[ChunkMeta], None]] = None,
                 lock: Optional[SpeakerLock] = None) -> None:
        self._user, self._cfg, self._store = user_id, config or Config(), store_dir
        self._on_unverified, self._on_meta = on_unverified, on_meta
        self._lock = lock
        self._enabled = True
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="speakerlock-pipecat")
        self._to16k = self._from16k = None

    async def start(self, sample_rate: int) -> None:
        if self._lock is None:
            loop = asyncio.get_running_loop()
            self._lock = await loop.run_in_executor(self._executor, lambda: SpeakerLock.create(
                self._cfg, self._user, self._store, on_unverified=self._on_unverified))
        if sample_rate != SAMPLE_RATE:
            import soxr  # noqa: PLC0415
            self._to16k = soxr.ResampleStream(sample_rate, SAMPLE_RATE, 1, dtype="float32")
            self._from16k = soxr.ResampleStream(SAMPLE_RATE, sample_rate, 1, dtype="float32")

    async def stop(self) -> None:
        if self._lock is not None:
            self._lock.close()
        self._executor.shutdown(wait=False)

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._enabled or self._lock is None:
            return audio
        x = from_pcm16(audio)
        if self._to16k is not None:
            x = self._to16k.resample_chunk(x)
        loop = asyncio.get_running_loop()
        y, meta = await loop.run_in_executor(self._executor, self._lock.process, x)
        if self._on_meta is not None:
            self._on_meta(meta)
        if self._from16k is not None:
            y = self._from16k.resample_chunk(y)
        return to_pcm16(y)
