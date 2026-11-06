"""Tests for Guard 6 — Action guard (Phase 2: full implementation)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_firewall._config import ActionConfig
from voice_firewall._types import ActionRequest, ActionVerdict, Decision, RiskLevel, TaggedSegment
from voice_firewall.guards.g6_action import ActionGuard, AuditLog, build_confirmation_text
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guard(
    confirm_high: bool = True,
    confirm_untrusted: bool = True,
    audit_path: str | None = None,
) -> ActionGuard:
    cfg = ActionConfig(
        confirm_high_risk=confirm_high,
        confirm_after_untrusted_content=confirm_untrusted,
        audit_log_path=audit_path,
    )
    return ActionGuard(cfg)


def _req(
    name: str,
    params: dict | None = None,
    untrusted: bool = False,
    segment: TaggedSegment | None = None,
) -> ActionRequest:
    return ActionRequest(
        name=name,
        params=params or {},
        segment=segment,
        untrusted_content_active=untrusted,
    )


def _dummy_segment(**kwargs) -> TaggedSegment:
    defaults = dict(audio=np.zeros(16000, dtype=np.float32), speaker_score=0.9)
    defaults.update(kwargs)
    return TaggedSegment(**defaults)


# ---------------------------------------------------------------------------
# Risk registry
# ---------------------------------------------------------------------------

class TestRiskRegistry:
    def test_known_high_risk(self) -> None:
        assert _guard().risk_of("send_email") == RiskLevel.HIGH

    def test_known_medium_risk(self) -> None:
        assert _guard().risk_of("type_text") == RiskLevel.MEDIUM

    def test_known_low_risk(self) -> None:
        assert _guard().risk_of("open_app") == RiskLevel.LOW

    def test_unknown_action_defaults_to_high(self) -> None:
        assert _guard().risk_of("completely_unknown") == RiskLevel.HIGH

    def test_all_configured_actions_resolve(self) -> None:
        cfg = ActionConfig()
        guard = ActionGuard(cfg)
        for level_name, actions in cfg.risk_levels.items():
            for action in actions:
                assert guard.risk_of(action) == RiskLevel(level_name)

    def test_custom_risk_config(self) -> None:
        cfg = ActionConfig(risk_levels={"high": ["my_dangerous"], "low": [], "medium": []})
        g = ActionGuard(cfg)
        assert g.risk_of("my_dangerous") == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# High-risk → CONFIRM
# ---------------------------------------------------------------------------

class TestHighRiskConfirmation:
    def test_high_risk_returns_confirm(self) -> None:
        d = _guard().authorize(_req("send_email", {"to": "a@b.com", "subject": "Hi"}))
        assert d.verdict == ActionVerdict.CONFIRM

    def test_high_risk_has_confirmation_text(self) -> None:
        d = _guard().authorize(_req("send_email", {"to": "a@b.com", "subject": "Hi"}))
        assert d.confirmation_text is not None
        assert len(d.confirmation_text) > 0

    def test_high_risk_text_contains_params(self) -> None:
        d = _guard().authorize(_req("send_email", {"to": "alice@example.com", "subject": "Secret"}))
        assert "alice@example.com" in d.confirmation_text
        assert "Secret" in d.confirmation_text

    def test_high_risk_delete_file(self) -> None:
        d = _guard().authorize(_req("delete_file", {"path": "/home/user/important.pdf"}))
        assert d.verdict == ActionVerdict.CONFIRM
        assert "/home/user/important.pdf" in d.confirmation_text

    def test_high_risk_make_payment(self) -> None:
        d = _guard().authorize(_req("make_payment", {"amount": 100, "to": "Bob"}))
        assert d.verdict == ActionVerdict.CONFIRM
        assert "100" in d.confirmation_text
        assert "Bob" in d.confirmation_text

    def test_high_risk_install_app(self) -> None:
        d = _guard().authorize(_req("install_app", {"app": "unknown_tool.exe"}))
        assert d.verdict == ActionVerdict.CONFIRM
        assert "unknown_tool.exe" in d.confirmation_text

    def test_high_risk_unknown_action_defaults_confirm(self) -> None:
        # Unknown actions default to HIGH → CONFIRM
        d = _guard().authorize(_req("launch_missile", {"target": "moon"}))
        assert d.verdict == ActionVerdict.CONFIRM

    def test_high_risk_disabled_allows(self) -> None:
        # When confirm_high_risk=False, high-risk actions are allowed
        d = _guard(confirm_high=False).authorize(_req("delete_file", {"path": "/tmp/x"}))
        assert d.verdict == ActionVerdict.ALLOW

    def test_audit_id_always_present(self) -> None:
        d = _guard().authorize(_req("open_app", {"app": "calc"}))
        assert d.audit_id != ""


# ---------------------------------------------------------------------------
# Low and medium risk (no untrusted content) → ALLOW
# ---------------------------------------------------------------------------

class TestLowMediumAllow:
    def test_low_risk_allowed(self) -> None:
        d = _guard().authorize(_req("open_app", {"app": "calc"}))
        assert d.verdict == ActionVerdict.ALLOW

    def test_medium_risk_allowed_without_untrusted(self) -> None:
        d = _guard().authorize(_req("type_text", {"text": "hello"}, untrusted=False))
        assert d.verdict == ActionVerdict.ALLOW

    def test_no_confirmation_text_for_allowed(self) -> None:
        d = _guard().authorize(_req("open_app", {}))
        assert d.confirmation_text is None


# ---------------------------------------------------------------------------
# Rule of Two — untrusted content flag
# ---------------------------------------------------------------------------

class TestRuleOfTwo:
    def test_medium_plus_untrusted_requires_confirm(self) -> None:
        d = _guard().authorize(_req("type_text", {"text": "pay bob"}, untrusted=True))
        assert d.verdict == ActionVerdict.CONFIRM

    def test_medium_untrusted_has_confirmation_text(self) -> None:
        d = _guard().authorize(_req("type_text", {"text": "pay bob"}, untrusted=True))
        assert d.confirmation_text is not None
        assert "pay bob" in d.confirmation_text

    def test_low_untrusted_still_allowed(self) -> None:
        # Low-risk actions are always trivial — Rule of Two only applies to MEDIUM+
        d = _guard().authorize(_req("open_app", {"app": "calc"}, untrusted=True))
        assert d.verdict == ActionVerdict.ALLOW

    def test_rule_of_two_reason_in_decision(self) -> None:
        d = _guard().authorize(_req("search_web", {"query": "bank account"}, untrusted=True))
        assert "Rule of Two" in d.reason or "untrusted" in d.reason.lower()

    def test_rule_of_two_disabled_allows_medium(self) -> None:
        d = _guard(confirm_untrusted=False).authorize(
            _req("type_text", {"text": "pay"}, untrusted=True)
        )
        assert d.verdict == ActionVerdict.ALLOW

    def test_high_plus_untrusted_still_confirms(self) -> None:
        # High risk is caught by Rule 1 before Rule 2 — both paths lead to CONFIRM
        d = _guard().authorize(_req("send_email", {"to": "x"}, untrusted=True))
        assert d.verdict == ActionVerdict.CONFIRM


# ---------------------------------------------------------------------------
# Source tags in Decision
# ---------------------------------------------------------------------------

class TestSourceTags:
    def test_source_tags_from_segment(self) -> None:
        seg = _dummy_segment(speaker_score=0.85, source="mic")
        d = _guard().authorize(_req("send_email", {"to": "x"}, segment=seg))
        assert d.source_tags["speaker_score"] == pytest.approx(0.85, abs=0.01)
        assert d.source_tags["source"] == "mic"

    def test_source_tags_empty_without_segment(self) -> None:
        d = _guard().authorize(_req("send_email", {"to": "x"}, segment=None))
        assert d.source_tags == {}

    def test_source_tags_loopback_suspect(self) -> None:
        seg = _dummy_segment(source="loopback-suspect", speaker_score=0.3)
        d = _guard().authorize(_req("delete_file", {"path": "/"}, segment=seg))
        assert d.source_tags["source"] == "loopback-suspect"

    def test_liveness_score_in_tags_when_present(self) -> None:
        seg = _dummy_segment(liveness_score=0.6)
        d = _guard().authorize(_req("send_email", {"to": "x"}, segment=seg))
        assert "liveness_score" in d.source_tags

    def test_liveness_score_absent_when_none(self) -> None:
        seg = _dummy_segment(liveness_score=None)
        d = _guard().authorize(_req("send_email", {"to": "x"}, segment=seg))
        assert "liveness_score" not in d.source_tags

    def test_transcript_flags_in_tags(self) -> None:
        seg = _dummy_segment(transcript_flags=["injection-suspect"])
        d = _guard().authorize(_req("send_email", {"to": "x"}, segment=seg))
        assert "injection-suspect" in d.source_tags.get("transcript_flags", [])


# ---------------------------------------------------------------------------
# Confirmation text generation
# ---------------------------------------------------------------------------

class TestConfirmationText:
    def test_send_email_template(self) -> None:
        t = build_confirmation_text("send_email", {"to": "alice@example.com", "subject": "Hi"})
        assert "alice@example.com" in t
        assert "Hi" in t

    def test_delete_file_template(self) -> None:
        t = build_confirmation_text("delete_file", {"path": "/data/secret.db"})
        assert "/data/secret.db" in t

    def test_unknown_action_generic(self) -> None:
        t = build_confirmation_text("my_custom_action", {"key": "value"})
        assert "my_custom_action" in t

    def test_no_params_no_crash(self) -> None:
        t = build_confirmation_text("send_email", {})
        assert "send_email" in t

    def test_make_payment_template(self) -> None:
        t = build_confirmation_text("make_payment", {"amount": 500, "to": "Alice"})
        assert "500" in t
        assert "Alice" in t

    def test_execute_command_template(self) -> None:
        t = build_confirmation_text("execute_command", {"command": "rm -rf /"})
        assert "rm -rf /" in t

    def test_text_is_string(self) -> None:
        t = build_confirmation_text("open_app", {"app": "calc"})
        assert isinstance(t, str)
        assert len(t) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_entry_written(self, tmp_path) -> None:
        log_path = str(tmp_path / "audit.jsonl")
        g = _guard(audit_path=log_path)
        g.authorize(_req("send_email", {"to": "a@b.com", "subject": "X"}))

        lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "send_email"
        assert entry["verdict"] == "confirm"
        assert entry["risk"] == "high"

    def test_audit_param_keys_not_values(self, tmp_path) -> None:
        # Audit log must NOT contain the raw parameter values (privacy)
        log_path = str(tmp_path / "audit.jsonl")
        g = _guard(audit_path=log_path)
        g.authorize(_req("send_email", {"to": "secret@example.com", "subject": "Secret body"}))

        raw = Path(log_path).read_text(encoding="utf-8")
        assert "secret@example.com" not in raw
        assert "Secret body" not in raw
        # But param keys are present
        assert "to" in raw
        assert "subject" in raw

    def test_audit_id_in_log(self, tmp_path) -> None:
        log_path = str(tmp_path / "audit.jsonl")
        g = _guard(audit_path=log_path)
        d = g.authorize(_req("open_app", {"app": "calc"}))

        entry = json.loads(Path(log_path).read_text(encoding="utf-8").strip())
        assert entry["audit_id"] == d.audit_id

    def test_audit_untrusted_flag_logged(self, tmp_path) -> None:
        log_path = str(tmp_path / "audit.jsonl")
        g = _guard(audit_path=log_path)
        g.authorize(_req("type_text", {"text": "pay"}, untrusted=True))

        entry = json.loads(Path(log_path).read_text(encoding="utf-8").strip())
        assert entry["untrusted"] is True

    def test_audit_accumulates_multiple_entries(self, tmp_path) -> None:
        log_path = str(tmp_path / "audit.jsonl")
        g = _guard(audit_path=log_path)
        for i in range(5):
            g.authorize(_req("open_app", {"app": f"app{i}"}))

        lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_no_audit_path_no_error(self) -> None:
        g = _guard(audit_path=None)
        # Must not raise
        g.authorize(_req("send_email", {"to": "x"}))

    def test_audit_creates_parent_dirs(self, tmp_path) -> None:
        log_path = str(tmp_path / "nested" / "dir" / "audit.jsonl")
        g = _guard(audit_path=log_path)
        g.authorize(_req("open_app", {}))
        assert Path(log_path).exists()


# ---------------------------------------------------------------------------
# Pipeline integration (fw.authorize passes untrusted_content_seen)
# ---------------------------------------------------------------------------

class TestPipelineAuthorize:
    def _make_fw(self):
        from voice_firewall._config import FirewallConfig
        from voice_firewall._pipeline import Firewall
        return Firewall(FirewallConfig.defaults())

    def test_authorize_allow_low_risk(self) -> None:
        fw = self._make_fw()
        d = fw.authorize("open_app", {"app": "calc"})
        assert d.verdict == ActionVerdict.ALLOW

    def test_authorize_confirm_high_risk(self) -> None:
        fw = self._make_fw()
        d = fw.authorize("send_email", {"to": "x"})
        assert d.verdict == ActionVerdict.CONFIRM

    def test_authorize_rule_of_two_wired(self) -> None:
        fw = self._make_fw()
        fw.mark_untrusted_content()
        d = fw.authorize("type_text", {"text": "pay bob"})
        assert d.verdict == ActionVerdict.CONFIRM

    def test_authorize_clear_untrusted_allows_medium(self) -> None:
        fw = self._make_fw()
        fw.mark_untrusted_content()
        fw.clear_untrusted_flag()
        d = fw.authorize("type_text", {"text": "hello"})
        assert d.verdict == ActionVerdict.ALLOW

    def test_authorize_returns_audit_id(self) -> None:
        fw = self._make_fw()
        d = fw.authorize("open_app", {})
        assert len(d.audit_id) > 0

    def test_authorize_with_segment(self) -> None:
        fw = self._make_fw()
        seg = _dummy_segment(speaker_score=0.9)
        d = fw.authorize("send_email", {"to": "x"}, segment=seg)
        assert d.source_tags.get("speaker_score") is not None
