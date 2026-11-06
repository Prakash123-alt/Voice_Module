"""WASAPI loopback capture: records everything the PC plays through its speakers.

This is the reference signal for the Self-sound guard (G2).  Unlike VoiceModule,
which only captures the assistant's TTS output, the firewall captures ALL PC audio
so that YouTube, calls, ads, and any other audio source are cancelled as well.

Platform isolation
------------------
Only Windows (WASAPI) is supported in Phase 1.  Other platforms get a ``StubLoopback``
that always returns silence — the guard still runs, but the AEC has no reference,
so it cannot suppress loopback audio.  The ``build_loopback()`` factory picks the
right implementation based on ``sys.platform``.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from voice_firewall._logging import log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LoopbackCapture(ABC):
    """Captures all PC speaker output as a reference stream."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the next frame (float32 @ 16 kHz), or None on timeout/silence."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """True when real loopback capture is active (not a stub)."""


# ---------------------------------------------------------------------------
# Stub (non-Windows / missing dependency)
# ---------------------------------------------------------------------------

class StubLoopback(LoopbackCapture):
    """Always returns silence.  Used on non-Windows or when sounddevice is missing.

    The Self-sound guard logs a warning when this is active so the user knows
    echo cancellation is degraded.
    """

    @property
    def available(self) -> bool:
        return False

    def start(self) -> None:
        log(logger, logging.WARNING,
            "loopback capture unavailable — self-sound guard running without reference "
            "(Windows WASAPI required)")

    def stop(self) -> None:
        pass

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        return None


# ---------------------------------------------------------------------------
# Windows WASAPI loopback
# ---------------------------------------------------------------------------

class WASAPILoopback(LoopbackCapture):
    """Capture all PC speaker output via WASAPI loopback using sounddevice.

    sounddevice on Windows exposes the WASAPI loopback device as a regular
    input device whose name contains the output device name and is marked as
    a loopback device.  We find it by looking for 'loopback' in the device
    hostapi name or by using the 'wasapi_exclusive' hostapi index.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 320,
        max_queue: int = 500,
    ) -> None:
        self._rate = sample_rate
        self._frame = frame_samples
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stream = None
        self._stopping = False
        self._device_index: Optional[int] = None

    @property
    def available(self) -> bool:
        return True

    def _find_loopback_device(self) -> Optional[int]:
        """Return the index of a WASAPI loopback capture device, or None."""
        import sounddevice as sd
        hostapis = sd.query_hostapis()
        wasapi_idx = next(
            (i for i, h in enumerate(hostapis) if "wasapi" in h["name"].lower()), None
        )
        if wasapi_idx is None:
            return None
        for i, d in enumerate(sd.query_devices()):
            if (
                d["hostapi"] == wasapi_idx
                and d["max_input_channels"] > 0
                and "loopback" in d["name"].lower()
            ):
                return i
        return None

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        # Mix to mono if stereo, resample inline if needed
        frame = indata.mean(axis=1).astype(np.float32) if indata.ndim > 1 else indata[:, 0].copy()
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass  # drop oldest is implicit (new frames overwrite); just discard

    def start(self) -> None:
        import sounddevice as sd
        self._stopping = False
        self._device_index = self._find_loopback_device()
        if self._device_index is None:
            log(logger, logging.WARNING,
                "WASAPI loopback device not found — self-sound guard has no reference")
            return
        self._stream = sd.InputStream(
            samplerate=self._rate,
            channels=1,
            dtype="float32",
            blocksize=self._frame,
            device=self._device_index,
            callback=self._callback,
        )
        self._stream.start()
        log(logger, logging.INFO, "loopback capture started",
            device=self._device_index, rate=self._rate)

    def stop(self) -> None:
        self._stopping = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log(logger, logging.WARNING, "loopback stop error", err=str(e))
            self._stream = None

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loopback(backend: str = "wasapi", **kwargs: object) -> LoopbackCapture:
    """Return the appropriate loopback implementation.

    ``backend`` values: ``"wasapi"`` (Windows only), ``"stub"`` (always silence).
    Unknown backends fall through to stub.
    """
    if backend == "stub":
        return StubLoopback()
    if backend == "wasapi":
        if sys.platform != "win32":
            log(logger, logging.WARNING,
                "loopback backend 'wasapi' requested on non-Windows — using stub")
            return StubLoopback()
        try:
            return WASAPILoopback(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            log(logger, logging.WARNING,
                "WASAPI loopback init failed — using stub", err=str(e))
            return StubLoopback()
    log(logger, logging.WARNING,
        "unknown loopback backend — using stub", backend=backend)
    return StubLoopback()
