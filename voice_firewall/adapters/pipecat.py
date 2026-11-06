"""Pipecat integration adapter (Phase 4).

Wraps the Voice Firewall as a Pipecat-compatible audio processor.
Pipecat uses an async frame-based pipeline; this adapter bridges
the synchronous Firewall.process() call.

Install Pipecat::
    pip install pipecat-ai

Usage::
    from voice_firewall.adapters.pipecat import PipecatFirewallProcessor
    from voice_firewall._config import FirewallConfig

    fw_processor = PipecatFirewallProcessor(FirewallConfig.from_yaml())
    # Add to your Pipecat Pipeline:
    # pipeline = Pipeline([..., fw_processor, stt_processor, ...])

Docs: https://docs.pipecat.ai/
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from voice_firewall._config import FirewallConfig
from voice_firewall._pipeline import Firewall
from voice_firewall._types import TaggedSegment

logger = logging.getLogger(__name__)


class PipecatFirewallProcessor:
    """Pipecat-compatible processor that guards audio frames.

    In Pipecat, processors receive ``AudioRawFrame`` objects and push them
    downstream.  This processor:
    1. Accumulates frames into utterance-length segments (via its own VAD).
    2. Passes complete segments through the Voice Firewall.
    3. Pushes the frame downstream only if the segment passes all guards.
    4. Drops the frame silently if any guard blocks it.

    If Pipecat is not installed, the class still imports cleanly but raises
    ``ImportError`` when ``process_frame()`` is called with a Pipecat frame.
    """

    def __init__(
        self,
        cfg: FirewallConfig,
        on_passed: Optional[callable] = None,
        on_dropped: Optional[callable] = None,
    ) -> None:
        self._fw = Firewall(cfg)
        self._on_passed = on_passed
        self._on_dropped = on_dropped

    def start(self) -> None:
        self._fw.start()

    def stop(self) -> None:
        self._fw.stop()

    async def process_frame(self, frame: object, direction: object = None) -> None:
        """Process a Pipecat AudioRawFrame through the firewall.

        Compatible with Pipecat's ``FrameProcessor.process_frame`` signature.
        """
        try:
            from pipecat.frames.frames import AudioRawFrame  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pipecat-ai is not installed. Run: pip install pipecat-ai"
            ) from exc

        if not isinstance(frame, AudioRawFrame):
            # Pass non-audio frames through unchanged (video, text, control, etc.)
            await self._push_frame(frame, direction)
            return

        # Convert Pipecat raw bytes → float32 numpy
        audio = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        tagged = self._fw.process(audio)

        if tagged is None:
            if self._on_dropped:
                self._on_dropped("dropped by guard pipeline")
            # Drop: do not push frame downstream
            return

        if self._on_passed:
            self._on_passed(tagged)
        # Pass the original frame downstream (guards do not modify the bytes)
        await self._push_frame(frame, direction)

    async def _push_frame(self, frame: object, direction: object) -> None:
        """Push frame to the next processor — override if subclassing."""
        # Base implementation: no-op (subclass or wire via Pipecat pipeline)
        pass

    @property
    def firewall(self) -> Firewall:
        return self._fw
