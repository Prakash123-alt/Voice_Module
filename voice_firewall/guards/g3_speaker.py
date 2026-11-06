"""Guard 3 — Speaker guard (Phase 1 Steps 4–6).

Responsibility
--------------
Verify that the speaker is the enrolled owner before passing the segment downstream.

Pipeline
--------
1. Embed the segment audio → 256-dim L2-normalised vector (run in this call;
   caller must NOT call from the DSP thread).
2. Push embedding into the rolling gate → cosine similarity vs profile centroid.
3. Decide: PASS | DROP based on threshold and mode.

Modes
-----
observe_mode (threshold is None)
    Gate logs scores but NEVER drops.  Mandatory until calibrate.py sets thresholds.

normal mode (threshold is set, assistant NOT speaking)
    score >= threshold → PASS
    score <  threshold → DROP

strict mode (threshold is set, assistant IS speaking — barge-in attempt)
    Uses strict_threshold (always >= threshold, never looser).
    score >= strict_threshold → PASS  (then Step 5 completes the duck/confirm flow)
    score <  strict_threshold → DROP

No enrolled profile
    Guard has no profile to compare against → PASS with warning tag.
    The firewall is insecure until enrollment is completed; this is logged clearly.

Fail-closed
    Any unexpected exception → DROP (inherited from GuardBase).
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional

import numpy as np

from voice_firewall._config import SpeakerConfig
from voice_firewall._logging import log
from voice_firewall._types import GuardResult, GuardVerdict, PipelineContext, RawSegment
from voice_firewall.calibration import ScoreLog
from voice_firewall.guards._base import GuardBase
from voice_firewall.speaker.embedder import SpeakerEmbedder, StubEmbedder, build_embedder
from voice_firewall.speaker.gate import GateResult, RollingEmbeddingGate, load_profile

logger = logging.getLogger(__name__)

# Embedding runs in a thread pool so it doesn't block the pipeline loop.
_EMBED_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="vfw-embed"
)


class SpeakerGuard(GuardBase):
    """Rolling cosine-similarity gate using WeSpeaker ResNet34 ONNX embeddings."""

    def __init__(
        self,
        cfg: SpeakerConfig,
        embedder: Optional[SpeakerEmbedder] = None,
        profile: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = cfg
        self._embedder = embedder if embedder is not None else self._load_embedder()
        self._profile = profile if profile is not None else load_profile(cfg.profile_path)
        self._gate: Optional[RollingEmbeddingGate] = None
        # Score log for observe-mode calibration data collection (optional)
        self._score_log: Optional[ScoreLog] = (
            ScoreLog(cfg.score_log_path) if cfg.score_log_path else None
        )

        if self._profile is not None:
            self._init_gate()
        else:
            log(logger, logging.WARNING,
                "speaker guard: no enrolled profile found",
                path=cfg.profile_path,
                hint="run `vfw-enroll` to enroll the owner's voice")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_embedder(self) -> SpeakerEmbedder:
        observe = self.cfg.observe_mode or self.cfg.threshold is None
        return build_embedder(
            model_name=self.cfg.onnx_model,
            offline=False,  # allow download; build_embedder falls back to Stub on error
        )

    def _init_gate(self) -> None:
        assert self._profile is not None
        threshold = self.cfg.threshold or 0.0
        strict = self.cfg.strict_threshold or threshold
        self._gate = RollingEmbeddingGate(
            profile=self._profile,
            threshold=threshold,
            strict_threshold=strict,
            observe_mode=self.cfg.observe_mode or self.cfg.threshold is None,
        )

    # ------------------------------------------------------------------
    # GuardBase interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "g3_speaker"

    def reset(self) -> None:
        if self._gate is not None:
            self._gate.reset()

    def _check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        # Case 1: No profile enrolled
        if self._profile is None or self._gate is None:
            log(logger, logging.WARNING,
                "speaker guard: passing without verification — no profile enrolled")
            return GuardResult(
                guard=self.name,
                verdict=GuardVerdict.PASS,
                reason="no profile — verification skipped",
                tags={
                    "speaker_score": None,
                    "enrolled": False,
                    "observe_mode": True,
                },
            )

        # Case 2: Segment too short — skip embedding, carry gate state forward
        if segment.speech_ms < self.cfg.min_segment_ms:
            log(logger, logging.DEBUG,
                "speaker guard: segment too short — skipping embedding",
                speech_ms=segment.speech_ms,
                min_ms=self.cfg.min_segment_ms)
            return GuardResult(
                guard=self.name,
                verdict=GuardVerdict.PASS,
                reason="segment too short — skipping",
                tags={"speaker_score": None, "enrolled": True,
                      "observe_mode": self._gate.observe_mode},
            )

        # Case 3: Embed (in a thread to keep call latency bounded;
        # futures.wait with a tight timeout degrades gracefully)
        try:
            future = _EMBED_POOL.submit(
                self._embedder.embed, segment.audio, 16000
            )
            embedding = future.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            log(logger, logging.ERROR, "speaker embedding timed out — dropping segment")
            return GuardResult(
                guard=self.name,
                verdict=GuardVerdict.DROP,
                reason="embedding timeout",
                tags={"speaker_score": None, "enrolled": True},
            )

        gate_result: GateResult = self._gate.push(
            embedding, assistant_speaking=ctx.assistant_speaking
        )

        score = round(gate_result.score, 4)
        threshold_used = gate_result.threshold_used
        observe = gate_result.observe_mode

        log(logger, logging.INFO,
            "speaker score",
            score=score,
            threshold=threshold_used,
            accepted=gate_result.accepted,
            observe_mode=observe,
            assistant_speaking=ctx.assistant_speaking)

        # Append to score log for calibration data collection (observe mode only).
        if self._score_log is not None and observe:
            try:
                self._score_log.append(score, label="unknown")
            except Exception as exc:  # noqa: BLE001
                log(logger, logging.WARNING, "score log write failed", err=str(exc))

        if gate_result.accepted:
            verdict = GuardVerdict.PASS
            reason = "pass" if not observe else "observe-pass"
        else:
            verdict = GuardVerdict.DROP
            reason = f"score {score:.4f} < threshold {threshold_used:.4f}"

        return GuardResult(
            guard=self.name,
            verdict=verdict,
            reason=reason,
            tags={
                "speaker_score": score,
                "threshold_used": threshold_used,
                "enrolled": True,
                "observe_mode": observe,
                "assistant_speaking": ctx.assistant_speaking,
            },
        )
