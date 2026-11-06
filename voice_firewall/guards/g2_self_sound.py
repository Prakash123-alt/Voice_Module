"""Guard 2 — Self-sound guard (full implementation, Phase 1 Step 3).

Responsibility
--------------
Detect audio that likely originated from the PC's speakers (YouTube, calls,
the assistant's own TTS voice) and tag it as ``source="loopback-suspect"``.

How it works
------------
The DSP loop runs WebRTC AEC3 frame-by-frame using the full WASAPI loopback
capture as the reference signal (all PC output, not only assistant TTS).
For each completed segment it records ``aec_suppression_db`` — the energy
reduction in dB that the AEC achieved.

This guard reads that field and decides:

  suppression_db >= loopback_suspect_db  →  source="loopback-suspect"  (PASS + tag)
  suppression_db  < loopback_suspect_db  →  source="mic"               (PASS)
  aec_suppression_db is None             →  loopback unavailable        (PASS + warning tag)

The guard NEVER drops audio by itself — detection quality depends on AEC
quality, and the AEC can produce false positives.  The Speaker guard (G3)
provides the actual gate.  G2 only annotates.

Fail-closed rule (from GuardBase): any unhandled exception → DROP.
"""
from __future__ import annotations

import logging

from voice_firewall._config import SelfSoundConfig
from voice_firewall._logging import log
from voice_firewall._types import GuardResult, GuardVerdict, PipelineContext, RawSegment
from voice_firewall.audio.spectral import near_ultrasound_ratio
from voice_firewall.guards._base import GuardBase

logger = logging.getLogger(__name__)


class SelfSoundGuard(GuardBase):
    """Tags segments whose AEC suppression exceeds the loopback-suspect threshold."""

    def __init__(self, cfg: SelfSoundConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "g2_self_sound"

    def _check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        # --- Case 1: loopback unavailable (StubLoopback active) ---
        if segment.aec_suppression_db is None:
            log(logger, logging.WARNING,
                "loopback capture unavailable — source cannot be verified",
                hint="install sounddevice and run on Windows for WASAPI loopback")
            return GuardResult(
                guard=self.name,
                verdict=GuardVerdict.PASS,
                reason="loopback unavailable — source unverified",
                tags={
                    "source": segment.source,
                    "loopback_available": False,
                    "aec_suppression_db": None,
                },
            )

        suppression = segment.aec_suppression_db
        suspect = suppression >= self.cfg.loopback_suspect_db

        if suspect:
            # Tag the segment in-place so downstream guards see the updated source.
            segment.source = "loopback-suspect"
            log(logger, logging.WARNING,
                "loopback-suspect audio detected",
                suppression_db=round(suppression, 1),
                threshold_db=self.cfg.loopback_suspect_db)

        source = "loopback-suspect" if suspect else "mic"

        # Near-ultrasound check on the mic audio (Phase 3).
        # Uses the top-band proxy: at 16 kHz this covers 6.4–8 kHz.
        near_us = near_ultrasound_ratio(segment.audio, 16000)
        if near_us >= self.cfg.near_ultrasound_threshold:
            segment.near_ultrasound_flag = True
            log(logger, logging.WARNING,
                "near-ultrasound energy detected in audio",
                ratio=round(near_us, 3),
                threshold=self.cfg.near_ultrasound_threshold)

        return GuardResult(
            guard=self.name,
            verdict=GuardVerdict.PASS,
            reason=f"source={source}",
            tags={
                "source": source,
                "loopback_available": True,
                "aec_suppression_db": round(suppression, 2),
                "near_ultrasound_ratio": round(near_us, 3),
            },
        )
