"""Base adapter interface for integrating the Voice Firewall with AI pipelines.

Usage pattern::

    class MyAdapter(FirewallAdapter):
        def on_segment_passed(self, segment: TaggedSegment) -> None:
            # forward audio to STT / LLM
            ...

        def on_segment_dropped(self, reason: str) -> None:
            # log or ignore
            ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from voice_firewall._config import FirewallConfig
from voice_firewall._pipeline import Firewall
from voice_firewall._types import Decision, TaggedSegment


class FirewallAdapter(ABC):
    """Base class for all Voice Firewall adapters.

    Subclass this and implement the two callbacks.  Call ``process_frame``
    from your audio pipeline's callback thread.
    """

    def __init__(self, cfg: FirewallConfig) -> None:
        self._fw = Firewall(cfg)

    @property
    def firewall(self) -> Firewall:
        """Access the underlying Firewall instance (e.g., for enrollment)."""
        return self._fw

    def start(self) -> None:
        self._fw.start()

    def stop(self) -> None:
        self._fw.stop()

    def process_frame(
        self,
        audio: np.ndarray,
        transcript: Optional[str] = None,
    ) -> Optional[TaggedSegment]:
        """Process one audio segment through all guards.

        Returns the TaggedSegment if it passed, or None if it was dropped.
        Calls ``on_segment_passed`` / ``on_segment_dropped`` automatically.
        """
        result = self._fw.process(audio, transcript=transcript)
        if result is None:
            self.on_segment_dropped("dropped by guard pipeline")
        else:
            self.on_segment_passed(result)
        return result

    def authorize(
        self,
        action_name: str,
        params: dict,
        segment: Optional[TaggedSegment] = None,
    ) -> Decision:
        """Authorize an action before execution. Always call this first."""
        return self._fw.authorize(action_name, params, segment)

    # ------------------------------------------------------------------
    # Callbacks to implement
    # ------------------------------------------------------------------

    @abstractmethod
    def on_segment_passed(self, segment: TaggedSegment) -> None:
        """Called when a segment passes all guards."""

    @abstractmethod
    def on_segment_dropped(self, reason: str) -> None:
        """Called when a segment is dropped by any guard."""
