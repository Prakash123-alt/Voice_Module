"""sounddevice-based audio I/O: 16 kHz mono, 20 ms frames, non-blocking callbacks
feeding thread-safe queues. The output keeps a tap of every frame it actually
renders so the AEC can use it as the far-end reference."""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Optional, Union

import numpy as np

from .interfaces import AudioInput, AudioOutput
from .logging_utils import log

logger = logging.getLogger(__name__)


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
    def __init__(self, sample_rate: int = 16000, frame_samples: int = 320, device=None, gain: float = 1.0,
                 max_queue: int = 500) -> None:
        self.sample_rate, self.frame_samples, self.device, self.gain = sample_rate, frame_samples, device, gain
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self._stream = None
        self._stopping = False
        self.dropped = 0
        self.error: Optional[Exception] = None

    def _callback(self, indata, frames, time_info, status) -> None:  # runs on PortAudio thread
        if status and status.input_overflow:
            self.dropped += 1
        frame = indata[:, 0].copy()
        if self.gain != 1.0:
            frame *= self.gain
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        import sounddevice as sd
        self._stopping = False
        self.error = None
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                      blocksize=self.frame_samples, device=_resolve_device(self.device, "input"),
                                      callback=self._callback, finished_callback=self._on_finished)
        self._stream.start()
        log(logger, logging.INFO, "mic started", device=self._stream.device, rate=self.sample_rate,
            frame=self.frame_samples)

    def _on_finished(self) -> None:
        if not self._stopping:
            self.error = RuntimeError("input stream finished unexpectedly (device unplugged?)")

    def stop(self) -> None:
        self._stopping = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:  # device may already be gone
                log(logger, logging.WARNING, "mic stop error", err=str(e))
            self._stream = None

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def active(self) -> bool:
        return self._stream is not None and self._stream.active


class SoundDeviceOutput(AudioOutput):
    """Pull-model output: callback drains a frame queue; silence when empty.
    `flush()` empties the queue so playback stops at the next callback (<= 20 ms)."""

    def __init__(self, sample_rate: int = 16000, frame_samples: int = 320, device=None, gain: float = 1.0) -> None:
        self.sample_rate, self.frame_samples, self.device, self.gain = sample_rate, frame_samples, device, gain
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._ref: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._pending = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._playing = threading.Event()
        self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._pending.size == 0:
                    try:
                        self._pending = self._q.get_nowait()
                    except queue.Empty:
                        break
                take = min(frames - filled, self._pending.size)
                out[filled:filled + take] = self._pending[:take]
                self._pending = self._pending[take:]
                filled += take
            still = filled > 0 or self._pending.size > 0 or not self._q.empty()
        if still:
            self._playing.set()
        else:
            self._playing.clear()
        if self.gain != 1.0:
            out *= self.gain
        outdata[:, 0] = out
        try:  # far-end reference for AEC (what actually hit the DAC)
            self._ref.put_nowait(out.copy())
        except queue.Full:
            pass

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                       blocksize=self.frame_samples, device=_resolve_device(self.device, "output"),
                                       callback=self._callback)
        self._stream.start()
        log(logger, logging.INFO, "speaker started", device=self._stream.device, rate=self.sample_rate)

    def stop(self) -> None:
        self.flush()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log(logger, logging.WARNING, "speaker stop error", err=str(e))
            self._stream = None

    def play(self, audio: np.ndarray) -> None:
        if audio.size:
            self._q.put(np.ascontiguousarray(audio, dtype=np.float32))
            self._playing.set()

    def flush(self) -> None:
        with self._lock:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._pending = np.zeros(0, dtype=np.float32)
        self._playing.clear()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def pop_reference(self) -> Optional[np.ndarray]:
        try:
            return self._ref.get_nowait()
        except queue.Empty:
            return None

    def reference_backlog(self) -> int:
        return self._ref.qsize()

    async def wait_until_done(self) -> None:
        while self.is_playing():
            await asyncio.sleep(0.02)
