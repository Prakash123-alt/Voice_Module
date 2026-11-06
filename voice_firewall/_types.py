"""Core data types shared across the entire firewall pipeline.

All types are plain dataclasses or enums — no business logic lives here.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk level assigned to an action or segment."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class GuardVerdict(str, Enum):
    """Per-guard decision for a single audio segment."""
    PASS = "pass"   # guard accepts the segment; pipeline continues
    DROP = "drop"   # guard rejects the segment; pipeline stops and drops it
    WARN = "warn"   # guard passes but attaches a warning tag


class ActionVerdict(str, Enum):
    """Final decision returned by fw.authorize()."""
    ALLOW   = "allow"    # proceed without user interaction
    CONFIRM = "confirm"  # show on-screen confirmation; wait for click/key press
    DENY    = "deny"     # block the action outright


# ---------------------------------------------------------------------------
# Pipeline primitives
# ---------------------------------------------------------------------------

@dataclass
class RawSegment:
    """An audio segment as it enters the guard pipeline.

    Produced by the VAD/segmenter. Guards annotate it progressively.
    Raw audio is float32, 16 kHz mono, values in [-1, 1].
    """
    audio: np.ndarray
    source: Literal["mic", "loopback-suspect"] = "mic"
    speech_ms: int = 0          # milliseconds of speech detected by the segmenter
    end_time: float = field(default_factory=time.perf_counter)  # perf_counter at utterance end
    # Filled by DSPLoop: energy suppression in dB produced by AEC over this segment.
    # None when loopback capture is unavailable (StubLoopback active).
    # Positive = AEC reduced energy; large positive → likely loopback audio.
    aec_suppression_db: float | None = None
    # Optional transcript provided by the caller or a preceding STT step.
    # G5 uses this for injection-pattern and speech-rate checks when present.
    transcript: str | None = None
    # Set by DSPLoop / G2 when near-ultrasound content is detected in the loopback
    # reference audio.  True → possible hidden-command attack via ultrasound.
    near_ultrasound_flag: bool = False


@dataclass
class GuardResult:
    """Record of one guard's decision for one segment."""
    guard: str                      # guard name, e.g. "g3_speaker"
    verdict: GuardVerdict
    reason: str = ""
    tags: dict[str, object] = field(default_factory=dict)
    latency_ms: float = 0.0        # time spent inside this guard


@dataclass
class TaggedSegment:
    """An audio segment that has passed all guards.

    This is what the AI agent receives. All fields are set by the guard that
    produces them; defaults are safe/conservative.
    """
    audio: np.ndarray
    transcript: str = ""

    # G3 — Speaker guard
    speaker_score: float = 0.0

    # G2 — Self-sound guard
    source: Literal["mic", "loopback-suspect"] = "mic"

    # G4 — Live-voice guard (None until Phase 4 model is loaded)
    liveness_score: float | None = None

    # G5 — Words guard
    transcript_flags: list[str] = field(default_factory=list)

    # G6 — Action guard / overall
    risk_level: RiskLevel = RiskLevel.LOW

    # Barge-in / interrupt flag
    # True  → segment arrived while the assistant was speaking AND passed the
    #         *strict* speaker threshold.  The caller should duck TTS and hand
    #         control to the user.
    # False → normal turn or unverified interrupt (drop TTS only if explicitly confirmed).
    interrupt: bool = False

    # Audit trail: one entry per guard that ran
    guard_trace: list[GuardResult] = field(default_factory=list)

    # Wall-clock identity for the audit log
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Action authorisation
# ---------------------------------------------------------------------------

@dataclass
class ActionRequest:
    """An action the AI agent wants to perform."""
    name: str                        # e.g. "send_email"
    params: dict[str, object]        # actual parameters — used to generate confirm text
    segment: TaggedSegment | None    # the speech that triggered this action (if any)
    # Set by the Firewall from PipelineContext.untrusted_content_seen before passing
    # to ActionGuard — ActionGuard must NOT read pipeline context directly.
    untrusted_content_active: bool = False


@dataclass
class Decision:
    """Result of fw.authorize(action, segment)."""
    verdict: ActionVerdict
    # Human-readable text shown in the confirmation dialog.
    # Generated from actual action params, NEVER from the AI's wording.
    confirmation_text: str | None = None
    reason: str = ""
    # UUID written to the audit log; returned to the caller for correlation.
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Provenance tags extracted from the triggering segment — shown in the UI
    # alongside the confirmation text so the user knows who spoke and what
    # the trust level was.  Empty dict when no segment is available.
    source_tags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline context (mutable, one per pipeline run)
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Mutable context passed to every guard's check() call.

    Guards read it to adjust thresholds; the pipeline updates it after key events.
    """
    assistant_speaking: bool = False
    # True after the AI has read content from an untrusted source (web, email, doc).
    # Triggers the Rule-of-Two: every non-trivial action requires confirmation.
    untrusted_content_seen: bool = False
