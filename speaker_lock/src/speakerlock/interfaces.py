"""Swappable model interfaces. Audio is always float32, 16 kHz, mono, shape (n,)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class VAD(ABC):
    """Frame-level voice activity detector (stateful, streaming)."""

    frame_samples: int

    @abstractmethod
    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for exactly `frame_samples` samples."""

    @abstractmethod
    def reset(self) -> None: ...


@dataclass
class TargetCondition:
    """Whatever an extractor needs to lock onto one speaker (computed once at enrollment)."""

    arrays: dict[str, np.ndarray]


class Extractor(ABC):
    """Causal target-speaker extractor. `process` accepts any number of samples and returns the isolated
    audio that is ready (possibly fewer samples); the total output equals the total input after `flush`."""

    #: algorithmic delay in samples (output sample k corresponds to input sample k, emitted this late)
    latency_samples: int = 0

    @abstractmethod
    def condition(self, sample: np.ndarray) -> TargetCondition:
        """Enrollment: compute the conditioning for a clean clip of the target (>= 3 s)."""

    @abstractmethod
    def start(self, cond: TargetCondition) -> None:
        """Begin a new stream locked to `cond` (resets internal state)."""

    @abstractmethod
    def process(self, chunk: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def flush(self) -> np.ndarray: ...


class Embedder(ABC):
    """Speaker embedding (d-vector / ECAPA style). Output is L2-normalised."""

    dim: int

    @abstractmethod
    def embed(self, audio: np.ndarray) -> np.ndarray: ...


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2norm(a), l2norm(b)))
