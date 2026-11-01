"""Re-blocking FIFO: accepts audio chunks of any size and hands out fixed-size frames."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class RingBuffer:
    """Preallocated circular float32 FIFO. Grows (doubling) only if a burst exceeds capacity."""

    def __init__(self, capacity: int = 16000) -> None:
        self._buf = np.zeros(max(1, capacity), np.float32)
        self._start = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32).ravel()
        n = x.shape[0]
        if n == 0:
            return
        if self._size + n > self._buf.shape[0]:
            self._grow(self._size + n)
        cap = self._buf.shape[0]
        end = (self._start + self._size) % cap
        first = min(n, cap - end)
        self._buf[end:end + first] = x[:first]
        self._buf[:n - first] = x[first:]
        self._size += n

    def pop(self, n: int) -> np.ndarray:
        if n > self._size:
            raise ValueError(f"only {self._size} samples buffered, asked for {n}")
        cap = self._buf.shape[0]
        idx = (self._start + np.arange(n)) % cap
        out = self._buf[idx].copy()
        self._start = (self._start + n) % cap
        self._size -= n
        return out

    def frames(self, n: int) -> Iterator[np.ndarray]:
        """Yield every complete n-sample frame currently buffered (remainder stays buffered)."""
        while self._size >= n:
            yield self.pop(n)

    def clear(self) -> None:
        self._start = self._size = 0

    def _grow(self, need: int) -> None:
        cap = self._buf.shape[0]
        while cap < need:
            cap *= 2
        data = self.pop(self._size) if self._size else np.zeros(0, np.float32)
        self._buf = np.zeros(cap, np.float32)
        self._start = 0
        self._buf[:data.shape[0]] = data
        self._size = data.shape[0]
