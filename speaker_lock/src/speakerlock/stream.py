"""Real-time runner: decouples the audio callback from processing.

`push()` is safe to call from a PortAudio / WebRTC callback (it only enqueues). A worker thread drains
*everything* queued and processes it in one call. When the CPU keeps up this is one frame per call (lowest
latency); when it is throttled (battery, thermal, other load), calls batch more audio, which amortises the
extractor's fixed per-call cost, so latency settles at a higher level instead of growing without bound.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE
from .pipeline import ChunkMeta, SpeakerLock

AudioSink = Callable[[np.ndarray, ChunkMeta], None]


class LiveRunner:
    def __init__(self, lock: SpeakerLock, sink: AudioSink, max_backlog_s: float = 2.0) -> None:
        self.lock, self.sink = lock, sink
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
        self._max_backlog = int(max_backlog_s * SAMPLE_RATE)
        self._backlog = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0                 # samples discarded because the backlog exceeded max_backlog_s
        self.e2e_ms: list[float] = []    # arrival of the newest sample -> its clean audio is handed to sink

    def start(self) -> "LiveRunner":
        self._thread = threading.Thread(target=self._loop, name="speakerlock-live", daemon=True)
        self._thread.start()
        return self

    def push(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, np.float32).ravel()
        self._backlog += chunk.shape[0]
        self._q.put((chunk, time.perf_counter()))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            items = [first]
            while True:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            audio = np.concatenate([c for c, _ in items])
            self._backlog -= audio.shape[0]
            if audio.shape[0] > self._max_backlog:          # hopelessly behind: keep the newest audio
                self.dropped += audio.shape[0] - self._max_backlog
                audio = audio[-self._max_backlog:]
            y, meta = self.lock.process(audio)
            arrived = items[-1][1]
            self.e2e_ms.append((time.perf_counter() - arrived) * 1000
                               + self.lock.latency_samples / SAMPLE_RATE * 1000)
            if y.shape[0]:
                self.sink(y, meta)

    def stop(self, flush: bool = True) -> None:
        while flush and not self._q.empty() and self._thread is not None and self._thread.is_alive():
            time.sleep(0.01)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if flush:
            y, meta = self.lock.flush()
            if y.shape[0]:
                self.sink(y, meta)
