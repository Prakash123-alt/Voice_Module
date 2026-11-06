"""Microphone capture via sounddevice.

Adapted from VoiceModule voice/audio_io.py (SoundDeviceInput, same author).
"""
from __future__ import annotations

import logging
import queue
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from voice_firewall._logging import log

logger = logging.getLogger(__name__)


class AudioInput(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Blocking read of one frame. Returns None on timeout."""


def _resolve_device(spec: Optional[Union[int, str]], kind: str) -> Optional[int]:
    if spec is None or isinstance(spec, int):
        return spec
    import sounddevice as sd
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if spec.lower() in d["name"].lower() and d[key] > 0:
            return i
    raise ValueError(f"No {kind} device matching {spec!r}")


class SoundDeviceInput(AudioInput):
    """Mic capture: PortAudio callback → thread-safe queue → read()."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 320,
        device: Optional[Union[int, str]] = None,
        gain: float = 1.0,
        max_queue: int = 500,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.gain = gain
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stream = None
        self._stopping = False
        self.dropped = 0
        self.error: Optional[Exception] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        frame = indata[:, 0].copy()
        if self.gain != 1.0:
            frame *= self.gain
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        import sounddevice as sd
        self._stopping = False
        self.error = None
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=_resolve_device(self.device, "input"),
            callback=self._callback,
            finished_callback=self._on_finished,
        )
        self._stream.start()
        log(logger, logging.INFO, "mic started",
            device=self._stream.device, rate=self.sample_rate, frame=self.frame_samples)

    def _on_finished(self) -> None:
        if not self._stopping:
            self.error = RuntimeError("input stream finished unexpectedly (device unplugged?)")

    def stop(self) -> None:
        self._stopping = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log(logger, logging.WARNING, "mic stop error", err=str(e))
            self._stream = None

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
