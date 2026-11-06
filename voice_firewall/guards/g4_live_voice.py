"""Guard 4 — Live-voice guard (Phase 4).

Produces an anti-spoofing / liveness score via an AASIST-style ONNX model.

IMPORTANT design rules (never relax these):
- This guard NEVER drops audio on its own.  It is a WARNING signal only.
- The liveness score is NEVER the sole proof of identity.
- A perfect voice clone will likely score near 1.0 — this is a known limit.
  The action guard (G6) is the safety net for high-risk actions.
- Claiming this guard "stops all deepfakes" is explicitly forbidden.

Score semantics:
  0.0 → almost certainly synthetic / spoofed
  1.0 → almost certainly a live human speaker
  0.5 → unknown (stub or model unavailable)

When score < warn_below, a "liveness-warning" tag is attached and risk is
escalated to HIGH. The segment is NOT dropped.
"""
from __future__ import annotations

import logging

from voice_firewall._config import LiveVoiceConfig
from voice_firewall._logging import log
from voice_firewall._types import GuardResult, GuardVerdict, PipelineContext, RawSegment
from voice_firewall.guards._base import GuardBase
from voice_firewall.liveness.model import AntispoofModel, StubAntispoofModel, build_model

logger = logging.getLogger(__name__)


class LiveVoiceGuard(GuardBase):
    """Anti-spoof / liveness score — WARNING signal only.

    Pass a pre-built ``AntispoofModel`` instance to inject a real model in tests.
    When ``cfg.model_path`` is set, ``build_model()`` loads an ONNX file.
    """

    def __init__(
        self,
        cfg: LiveVoiceConfig,
        model: AntispoofModel | None = None,
    ) -> None:
        self.cfg = cfg
        self._model: AntispoofModel = model if model is not None else build_model(cfg.model_path)
        if cfg.enabled and not self._model.available:
            log(logger, logging.WARNING,
                "LiveVoiceGuard enabled but running with StubAntispoofModel — "
                "scores are 0.5 (unknown). Configure model_path for real detection.")

    @property
    def name(self) -> str:
        return "g4_live_voice"

    def _check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        if not self.cfg.enabled:
            return GuardResult(
                guard=self.name,
                verdict=GuardVerdict.PASS,
                reason="disabled in config",
                tags={"liveness_score": None},
            )

        score = self._model.score(segment.audio, sample_rate=16000)
        is_suspect = score < self.cfg.warn_below
        tags: dict = {
            "liveness_score": round(score, 4),
            "model_available": self._model.available,
        }

        if is_suspect:
            tags["liveness_warning"] = True
            log(logger, logging.WARNING,
                "liveness WARNING — possible synthetic voice",
                score=round(score, 4),
                threshold=self.cfg.warn_below,
                model_available=self._model.available)

        return GuardResult(
            guard=self.name,
            verdict=GuardVerdict.PASS,          # never drops
            reason=f"liveness={score:.4f}" + (" [WARNING]" if is_suspect else ""),
            tags=tags,
        )
