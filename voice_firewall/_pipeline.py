"""Firewall — wires all six guards into a sequential pipeline.

process(frame) is called on every 20 ms audio frame from the DSP thread.
authorize(action, segment) is called by the AI agent before executing any action.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

from voice_firewall._config import FirewallConfig
from voice_firewall._logging import Metrics, log
from voice_firewall._types import (
    ActionRequest,
    Decision,
    GuardVerdict,
    PipelineContext,
    RawSegment,
    TaggedSegment,
)
from voice_firewall.guards.g1_entry import EntryGuard
from voice_firewall.guards.g2_self_sound import SelfSoundGuard
from voice_firewall.guards.g3_speaker import SpeakerGuard
from voice_firewall.guards.g4_live_voice import LiveVoiceGuard
from voice_firewall.guards.g5_words import WordsGuard
from voice_firewall.guards.g6_action import ActionGuard

logger = logging.getLogger(__name__)


class Firewall:
    """Public entry point: process() and authorize().

    Instantiate once per session:
        fw = Firewall(FirewallConfig.from_yaml("firewall.yaml"))
        fw.start()

    Process audio frames (call from DSP thread or main loop):
        tagged = fw.process(audio_frame)   # None if dropped

    Authorize before any AI action:
        decision = fw.authorize("send_email", {"to": "...", "body": "..."}, tagged)
    """

    def __init__(
        self,
        cfg: FirewallConfig,
        on_barge_in: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = cfg
        self._metrics = Metrics(cfg.metrics.ring_buffer_size)
        self._ctx = PipelineContext()

        # Optional callback fired when a segment arrives while the assistant is
        # speaking — BEFORE guards run.  Use it to duck TTS volume immediately.
        # The interrupt is confirmed only if the segment passes G3 (strict threshold).
        self._on_barge_in: Optional[Callable[[], None]] = on_barge_in

        # Guards (in pipeline order)
        self._g1 = EntryGuard()
        self._g2 = SelfSoundGuard(cfg.guards.self_sound)
        self._g3 = SpeakerGuard(cfg.guards.speaker)
        self._g4 = LiveVoiceGuard(cfg.guards.live_voice)
        self._g5 = WordsGuard(cfg.guards.words)
        self._g6_action = ActionGuard(cfg.guards.action)

        self._guards = [self._g1, self._g2, self._g3, self._g4, self._g5]
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._started = True
        log(logger, logging.INFO, "firewall started",
            guards=[g.name for g in self._guards])

    def stop(self) -> None:
        self._started = False
        for g in self._guards:
            g.reset()
        log(logger, logging.INFO, "firewall stopped",
            **{k: v["p50_ms"] for k, v in self._metrics.summary().items()})

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        transcript: Optional[str] = None,
    ) -> Optional[TaggedSegment]:
        """Run all guards on a completed audio segment.

        Returns a TaggedSegment if the audio passes all guards, None if dropped.
        Call this once per utterance (after VAD segments the mic stream).

        Parameters
        ----------
        audio:
            Float32 16 kHz mono audio of the completed utterance.
        transcript:
            Optional pre-computed transcript.  When provided, G5 (Words guard)
            runs injection-pattern and speech-rate checks against it.

        Barge-in flow (when assistant is speaking):
          1. Duck  — ``on_barge_in`` callback fired immediately (before guard chain).
          2. Verify — G3 runs with the strict speaker threshold.
          3. Confirm — if G3 passes → TaggedSegment.interrupt = True (caller stops TTS).
                       if G3 drops  → segment discarded (TTS continues).
        """
        t_start = time.perf_counter()
        was_assistant_speaking = self._ctx.assistant_speaking
        segment = RawSegment(audio=audio, transcript=transcript)
        trace = []

        # Step 1 (Duck): fire the barge-in callback before the guard chain.
        if was_assistant_speaking and self._on_barge_in is not None:
            try:
                self._on_barge_in()
            except Exception as exc:  # noqa: BLE001
                log(logger, logging.WARNING, "on_barge_in callback raised",
                    err=str(exc))

        for guard in self._guards:
            with self._metrics.timer(guard.name):
                result = guard.check(segment, self._ctx)
            trace.append(result)
            if result.verdict == GuardVerdict.DROP:
                log(logger, logging.INFO, "segment dropped",
                    guard=guard.name, reason=result.reason,
                    was_barge_in=was_assistant_speaking)
                return None

        # Build the TaggedSegment from guard trace results
        tagged = self._build_tagged(segment, trace, was_assistant_speaking)

        total_ms = (time.perf_counter() - t_start) * 1000
        self._metrics.record("pipeline_total", total_ms)
        self._metrics.check_latency_budget(
            logger, self.cfg.metrics.target_latency_ms, "pipeline_total"
        )
        log(logger, logging.DEBUG, "segment passed all guards",
            total_ms=round(total_ms, 1),
            speaker_score=round(tagged.speaker_score, 3))
        return tagged

    def _build_tagged(self, segment: RawSegment, trace: list, was_barge_in: bool = False) -> TaggedSegment:
        """Assemble TaggedSegment from accumulated guard results."""
        speaker_score = 0.0
        liveness_score = None
        transcript_flags: list[str] = []

        for result in trace:
            tags = result.tags or {}
            if "speaker_score" in tags and tags["speaker_score"] is not None:
                speaker_score = float(tags["speaker_score"])
            if "liveness_score" in tags and tags["liveness_score"] is not None:
                liveness_score = float(tags["liveness_score"])
            if "transcript_flags" in tags:
                transcript_flags.extend(tags["transcript_flags"])

        # Step 3 (Confirm): a barge-in is confirmed when the segment was produced
        # while the assistant was speaking AND it passed G3's strict threshold.
        # The caller reads TaggedSegment.interrupt and stops TTS playback.
        g3_passed_strict = was_barge_in and any(
            r.guard == "g3_speaker" and r.verdict.value == "pass"
            and r.tags.get("assistant_speaking") is True
            for r in trace
        )

        return TaggedSegment(
            audio=segment.audio,
            source=segment.source,
            speaker_score=speaker_score,
            liveness_score=liveness_score,
            transcript_flags=transcript_flags,
            interrupt=g3_passed_strict,
            guard_trace=trace,
        )

    # ------------------------------------------------------------------
    # Action authorisation
    # ------------------------------------------------------------------

    def authorize(
        self,
        action_name: str,
        params: dict,
        segment: Optional[TaggedSegment] = None,
    ) -> Decision:
        """Decide whether the AI agent may execute an action.

        Always call this before performing any action on behalf of the user.
        """
        request = ActionRequest(
            name=action_name,
            params=params,
            segment=segment,
            untrusted_content_active=self._ctx.untrusted_content_seen,
        )
        decision = self._g6_action.authorize(request)
        log(logger, logging.INFO, "action authorised",
            action=action_name, verdict=decision.verdict.value,
            audit_id=decision.audit_id)
        return decision

    # ------------------------------------------------------------------
    # Context control (called by integrations)
    # ------------------------------------------------------------------

    def set_assistant_speaking(self, speaking: bool) -> None:
        """Notify the firewall when the assistant starts/stops speaking.

        This tightens the speaker threshold during barge-in attempts.
        """
        self._ctx.assistant_speaking = speaking

    def mark_untrusted_content(self) -> None:
        """Call when the AI reads content from an untrusted source.

        Activates the Rule of Two: subsequent actions require confirmation.
        """
        self._ctx.untrusted_content_seen = True
        log(logger, logging.WARNING, "untrusted content flag set — Rule of Two active")

    def clear_untrusted_flag(self) -> None:
        """Reset after the user explicitly acknowledges the untrusted content."""
        self._ctx.untrusted_content_seen = False

    def set_reference_audio(self, frame: np.ndarray) -> None:
        """Feed assistant TTS output to the AEC reference buffer.

        Called by integrations that manage playback directly (not via loopback).
        """
        # Phase 1 Step 3: wire to SelfSoundGuard's AEC.
        pass

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics_summary(self) -> dict:
        return self._metrics.summary()
