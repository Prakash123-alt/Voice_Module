"""Guard 5 — Words guard (Phase 3).

Checks the transcript and audio for suspicious content. Never drops audio;
only attaches flags to the TaggedSegment for downstream use by G6.

Flags added to transcript_flags:
  empty-transcript               — speech detected but transcript empty/whitespace
  injection:<pattern>            — known injection phrasing detected
  rate-too-fast:<N>wpm           — word rate implausibly high (machine-generated?)
  rate-too-slow:<N>wpm           — word rate implausibly low
  dual-transcribe-mismatch:<N>   — two STT runs diverged (optional, requires backend)
  dual-transcribe-error          — dual-transcription raised an exception
  background-audio-suspect       — energy outside speech band is anomalously high
  near-ultrasound-detected       — loopback reference had near-ultrasound energy
"""
from __future__ import annotations

import logging
from typing import Optional

from voice_firewall._config import WordsConfig
from voice_firewall._logging import log
from voice_firewall._types import GuardResult, GuardVerdict, PipelineContext, RawSegment
from voice_firewall.audio.spectral import background_energy_ratio
from voice_firewall.guards._base import GuardBase
from voice_firewall.words.dual_transcribe import DualTranscriber, StubDualTranscriber, check_dual_transcribe
from voice_firewall.words.injection import check_injection
from voice_firewall.words.rate import check_rate_mismatch

logger = logging.getLogger(__name__)


class WordsGuard(GuardBase):
    """Transcript sanity and injection-pattern checks.

    Pass a custom ``DualTranscriber`` instance for dual-transcription support;
    defaults to ``StubDualTranscriber`` (always passes, no STT required).
    """

    def __init__(
        self,
        cfg: WordsConfig,
        dual_transcriber: Optional[DualTranscriber] = None,
    ) -> None:
        self.cfg = cfg
        self._dual: DualTranscriber = dual_transcriber or StubDualTranscriber()

    @property
    def name(self) -> str:
        return "g5_words"

    def _check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        flags: list[str] = []

        transcript = segment.transcript or ""

        # 1. Empty / garbled transcript
        if self.cfg.flag_empty_transcript:
            rate_flags = check_rate_mismatch(transcript, segment.speech_ms)
            flags.extend(rate_flags)

        # 2. Injection patterns in transcript
        if self.cfg.flag_injection_patterns and transcript:
            inject_flags = check_injection(transcript)
            flags.extend(inject_flags)

        # 3. Dual transcription (only when explicitly enabled and transcript exists)
        if self.cfg.dual_transcribe_on_high_risk and transcript:
            dt_flags = check_dual_transcribe(self._dual, segment.audio)
            flags.extend(dt_flags)

        # 4. Background audio injection — spectral check on mic audio
        if self.cfg.background_inject_threshold < 1.0:
            bg_ratio = background_energy_ratio(segment.audio, 16000)
            if bg_ratio >= self.cfg.background_inject_threshold:
                flags.append(f"background-audio-suspect:{bg_ratio:.2f}")

        # 5. Propagate near-ultrasound flag from DSP/G2
        if segment.near_ultrasound_flag:
            flags.append("near-ultrasound-detected")

        if flags:
            log(logger, logging.WARNING, "words guard: flags raised",
                flags=flags, transcript_len=len(transcript))

        return GuardResult(
            guard=self.name,
            verdict=GuardVerdict.PASS,  # never drops; only annotates
            reason="pass" if not flags else f"{len(flags)} flag(s) raised",
            tags={"transcript_flags": flags},
        )
