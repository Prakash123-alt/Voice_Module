"""Tests for _logging.py: secret redaction and bounded ring-buffer metrics."""
from __future__ import annotations

import logging

import pytest

from voice_firewall._logging import Metrics, redact, setup_logging


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

class TestRedact:
    def test_openrouter_key_masked(self) -> None:
        line = "Authorization: Bearer sk-or-v1-abc123XYZsecret"
        out = redact(line)
        assert "sk-or-v1-abc123XYZsecret" not in out
        assert "***" in out

    def test_bearer_token_masked(self) -> None:
        out = redact("header: Bearer eyJhbGciOiJIUzI1NiJ9.secret.sig")
        assert "eyJhbGciOiJIUzI1NiJ9.secret.sig" not in out
        assert "***" in out

    def test_hf_token_masked(self) -> None:
        out = redact("token=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcde")
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in out
        assert "***" in out

    def test_safe_text_unchanged(self) -> None:
        line = "speaker score=0.81 speech_ms=1240"
        assert redact(line) == line

    def test_empty_string(self) -> None:
        assert redact("") == ""

    def test_multiple_secrets_in_one_line(self) -> None:
        line = "key=sk-or-v1-longkeysecret token=hf_SomeHFToken1234567890"
        out = redact(line)
        assert "longkeysecret" not in out
        assert "SomeHFToken" not in out

    def test_no_false_positive_on_sk_prefix(self) -> None:
        # "sk-" that is too short should not crash
        out = redact("sk-short")
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Bounded ring-buffer metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_ring_buffer_bounded(self) -> None:
        m = Metrics(ring_buffer_size=10)
        for i in range(50):
            m.record("stage_a", float(i))
        summary = m.summary()
        # Must never exceed maxlen
        assert summary["stage_a"]["n"] == 10

    def test_p50_correct(self) -> None:
        m = Metrics(ring_buffer_size=100)
        for v in range(1, 11):  # 1..10
            m.record("lat", float(v))
        s = m.summary()["lat"]
        # Sorted [1,2,3,4,5,6,7,8,9,10]; index 5 → 6
        assert s["p50_ms"] == 6.0

    def test_p95_does_not_exceed_max(self) -> None:
        m = Metrics(ring_buffer_size=200)
        for v in range(100):
            m.record("x", float(v))
        s = m.summary()["x"]
        assert s["p95_ms"] <= 99.0

    def test_empty_stage_summary(self) -> None:
        m = Metrics()
        s = m.summary().get("nonexistent")
        assert s is None

    def test_timer_context_manager(self) -> None:
        import time
        m = Metrics()
        with m.timer("work"):
            time.sleep(0.005)
        s = m.summary()["work"]
        assert s["n"] == 1
        assert s["last_ms"] >= 4.0  # at least 4 ms

    def test_multiple_stages(self) -> None:
        m = Metrics(ring_buffer_size=50)
        m.record("a", 1.0)
        m.record("b", 2.0)
        summary = m.summary()
        assert "a" in summary
        assert "b" in summary
        assert summary["a"]["n"] == 1
        assert summary["b"]["n"] == 1

    def test_latency_budget_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        m = Metrics()
        m.record("pipeline_total", 999.0)
        logger = logging.getLogger("test_budget")
        with caplog.at_level(logging.WARNING):
            m.check_latency_budget(logger, target_ms=150.0, stage="pipeline_total")
        assert any("latency budget exceeded" in r.message for r in caplog.records)

    def test_latency_budget_no_warning_when_ok(self, caplog: pytest.LogCaptureFixture) -> None:
        m = Metrics()
        m.record("pipeline_total", 50.0)
        logger = logging.getLogger("test_budget_ok")
        with caplog.at_level(logging.WARNING):
            m.check_latency_budget(logger, target_ms=150.0, stage="pipeline_total")
        assert not any("latency budget exceeded" in r.message for r in caplog.records)
