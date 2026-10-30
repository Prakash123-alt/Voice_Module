"""Abstract interfaces. Every pipeline stage is swappable by implementing one of these.

Audio convention everywhere: numpy float32 arrays in [-1, 1], 16 kHz mono,
unless an implementation explicitly documents otherwise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np


@dataclass
class VerificationResult:
    accepted: bool
    score: float
    threshold: float


class AudioInput(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Blocking read of one frame (frame_samples float32). None on timeout."""


class AudioOutput(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def play(self, audio: np.ndarray) -> None:
        """Enqueue audio (float32 @ 16 kHz) for playback; returns immediately."""

    @abstractmethod
    def flush(self) -> None:
        """Drop everything queued; playback stops within one frame (<= 20 ms)."""

    @abstractmethod
    def is_playing(self) -> bool: ...

    @abstractmethod
    def pop_reference(self) -> Optional[np.ndarray]:
        """Next frame that was actually sent to the DAC (AEC far-end reference), or None."""

    @abstractmethod
    async def wait_until_done(self) -> None: ...


class AECProcessor(ABC):
    available: bool = True

    @abstractmethod
    def process(self, mic: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
        """Return mic with the (delayed) reference echo removed."""

    def reset(self) -> None: ...


class Denoiser(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class VADDetector(ABC):
    @abstractmethod
    def probability(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for this frame."""

    def reset(self) -> None: ...


class SpeakerVerifier(ABC):
    @abstractmethod
    def embed(self, audio: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def verify(self, audio: np.ndarray, min_samples: Optional[int] = None) -> VerificationResult:
        """Accept iff cosine(embed(audio), profile) >= threshold. Audio shorter than
        `min_samples` (default: implementation's own minimum) is rejected."""


class STTEngine(ABC):
    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> str: ...


class LLMClient(ABC):
    @abstractmethod
    def stream_chat(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Yield text deltas. Must stop promptly when the consumer cancels."""

    async def aclose(self) -> None: ...


class TTSEngine(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield float32 @ 16 kHz chunks for `text`."""
