"""Guard 6 — Action guard (Phase 2).

Responsibility:
- Maintain a risk registry (low / medium / high per action name).
- HIGH risk → always CONFIRM (on-screen dialog, click/key required).
- Rule of Two: after untrusted content (web page, email, document), every
  non-trivial (medium+) action requires CONFIRM.
- Confirmation text is generated from ACTUAL action parameters, never from
  the AI agent's wording.
- Audit-log every action, decision, and confirmation to a JSONL file.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from voice_firewall._config import ActionConfig
from voice_firewall._logging import log, redact
from voice_firewall._types import (
    ActionRequest,
    ActionVerdict,
    Decision,
    RiskLevel,
    TaggedSegment,
)

logger = logging.getLogger(__name__)

# Risk level ordinal for comparison
_RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}

# ---------------------------------------------------------------------------
# Confirmation text generation
# ---------------------------------------------------------------------------

# Templates for known actions. Keys must match the action names in ActionConfig.
# Values use str.format(**params) — only params present in the template are used.
_CONFIRM_TEMPLATES: dict[str, str] = {
    "send_message":        "Send message to '{to}': {body!r}",
    "send_email":          "Send email to '{to}' with subject '{subject}'",
    "make_payment":        "Make payment of {amount} to '{to}'",
    "delete_file":         "Delete file: '{path}'",
    "install_app":         "Install application: '{app}'",
    "change_settings":     "Change setting '{setting}' to '{value}'",
    "execute_command":     "Execute command: '{command}'",
    "access_credentials":  "Access credentials for: '{service}'",
    "type_text":           "Type text: {text!r}",
    "search_web":          "Search the web for: '{query}'",
    "copy_to_clipboard":   "Copy to clipboard: {content!r}",
    "open_app":            "Open application: '{app}'",
    "navigate":            "Navigate to: '{url}'",
    "read_text":           "Read text from: '{source}'",
}


def build_confirmation_text(name: str, params: dict) -> str:
    """Generate a human-readable confirmation prompt from action name and actual params.

    Uses per-action templates for known actions; falls back to a generic format.
    Text is generated from actual params — NEVER from the AI's wording.
    """
    template = _CONFIRM_TEMPLATES.get(name)
    if template:
        try:
            # Only use keys that exist in params; missing template vars fall through
            str_params = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                          for k, v in params.items()}
            return template.format(**str_params)
        except (KeyError, ValueError):
            pass  # fall through to generic

    if params:
        param_parts = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"Perform action '{name}' with: {param_parts}"
    return f"Perform action '{name}'"


def _extract_source_tags(segment: Optional[TaggedSegment]) -> dict:
    """Pull provenance fields out of a TaggedSegment for display in the UI."""
    if segment is None:
        return {}
    tags: dict = {
        "speaker_score": round(segment.speaker_score, 3),
        "source": segment.source,
    }
    if segment.liveness_score is not None:
        tags["liveness_score"] = round(segment.liveness_score, 3)
    if segment.transcript_flags:
        tags["transcript_flags"] = list(segment.transcript_flags)
    tags["segment_id"] = segment.segment_id
    return tags


# ---------------------------------------------------------------------------
# Audit log (JSONL, thread-safe)
# ---------------------------------------------------------------------------

class AuditLog:
    """Thread-safe JSONL audit log.

    One entry per authorization call. Entries contain action name, verdict,
    risk level, audit_id, source tags, and param keys (never raw param values,
    to avoid leaking sensitive data).

    Format (one JSON object per line)::

        {"ts": 1234567890.123, "audit_id": "...", "action": "send_email",
         "verdict": "confirm", "risk": "high", "param_keys": ["to", "subject"],
         "untrusted": false, "speaker_score": 0.91, "source": "mic"}
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, entry: dict) -> None:
        line = redact(json.dumps(entry, default=str))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------------------
# ActionGuard — full Phase 2 implementation
# ---------------------------------------------------------------------------

class ActionGuard:
    """Risk-level registry + confirmation gate + audit log.

    Not a GuardBase subclass — it operates on *actions*, not audio segments.
    Called via fw.authorize(), not the per-segment pipeline.
    """

    def __init__(self, cfg: ActionConfig) -> None:
        self.cfg = cfg
        # Build reverse lookup: action_name → RiskLevel
        self._risk: dict[str, RiskLevel] = {}
        for level_name, actions in cfg.risk_levels.items():
            level = RiskLevel(level_name)
            for action in actions:
                self._risk[action] = level
        # Audit log (optional — None if audit_log_path is not configured)
        self._audit: Optional[AuditLog] = (
            AuditLog(cfg.audit_log_path) if cfg.audit_log_path else None
        )

    # ------------------------------------------------------------------
    # Risk registry
    # ------------------------------------------------------------------

    def risk_of(self, action_name: str) -> RiskLevel:
        """Return the configured risk level, defaulting to HIGH for unknowns.

        Unknown actions are assumed high-risk (fail-closed).
        """
        return self._risk.get(action_name, RiskLevel.HIGH)

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def authorize(self, request: ActionRequest) -> Decision:
        """Decide whether to ALLOW, CONFIRM, or DENY an action.

        Decision logic (in priority order):
        1. HIGH risk + confirm_high_risk → CONFIRM
        2. Rule of Two: untrusted content active + non-trivial (MEDIUM+) → CONFIRM
        3. All other → ALLOW

        Confirmation text is always generated from actual params.
        Every call is written to the audit log.
        """
        risk = self.risk_of(request.name)
        source_tags = _extract_source_tags(request.segment)
        audit_id = str(uuid.uuid4())

        verdict, reason, confirmation_text = self._decide(request, risk)

        decision = Decision(
            verdict=verdict,
            confirmation_text=confirmation_text,
            reason=reason,
            audit_id=audit_id,
            source_tags=source_tags,
        )

        self._log_decision(request, risk, decision)
        return decision

    def _decide(
        self, request: ActionRequest, risk: RiskLevel
    ) -> tuple[ActionVerdict, str, Optional[str]]:
        """Core decision logic. Returns (verdict, reason, confirmation_text)."""

        # Rule 1 — HIGH risk always requires confirmation (if configured)
        if risk == RiskLevel.HIGH and self.cfg.confirm_high_risk:
            text = build_confirmation_text(request.name, request.params)
            return (
                ActionVerdict.CONFIRM,
                f"high-risk action requires confirmation",
                text,
            )

        # Rule 2 — Rule of Two: untrusted content + non-trivial action
        if (
            request.untrusted_content_active
            and self.cfg.confirm_after_untrusted_content
            and _RISK_RANK[risk] >= _RISK_RANK[RiskLevel.MEDIUM]
        ):
            text = build_confirmation_text(request.name, request.params)
            return (
                ActionVerdict.CONFIRM,
                "Rule of Two: untrusted content active — confirmation required",
                text,
            )

        # Rule 3 — ALLOW
        return ActionVerdict.ALLOW, f"risk={risk.value} — allowed", None

    # ------------------------------------------------------------------
    # Logging + audit
    # ------------------------------------------------------------------

    def _log_decision(
        self, request: ActionRequest, risk: RiskLevel, decision: Decision
    ) -> None:
        log(logger, logging.INFO, "action decision",
            action=request.name,
            verdict=decision.verdict.value,
            risk=risk.value,
            untrusted=request.untrusted_content_active,
            audit_id=decision.audit_id)

        if self._audit is not None:
            entry = {
                "ts": round(time.time(), 3),
                "audit_id": decision.audit_id,
                "action": request.name,
                "verdict": decision.verdict.value,
                "risk": risk.value,
                "param_keys": sorted(request.params.keys()),
                "untrusted": request.untrusted_content_active,
                **decision.source_tags,
            }
            try:
                self._audit.append(entry)
            except Exception as exc:  # noqa: BLE001
                log(logger, logging.WARNING, "audit log write failed", err=str(exc))
