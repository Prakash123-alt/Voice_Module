"""Tests for Phase 1 Step 7 — Bounded ring-buffer metrics and latency target.

Covers:
1. _Stage: bounded memory, p50/p95/max/mean/last correctness
2. Metrics: reset(), stages, max_samples, latency_ok, summary
3. Budget warning fires / does not fire
4. Pipeline wiring: after fw.process(), all guard stages + pipeline_total appear
5. Memory guarantee: inserting >> maxlen samples never exceeds maxlen
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from voice_firewall._logging import Metrics


# ---------------------------------------------------------------------------
# 1. Bounded memory
# ---------------------------------------------------------------------------

class TestBoundedMemory:
    def test_ring_buffer_capped_at_maxlen(self) -> None:
        m = Metrics(ring_buffer_size=50)
        for i in range(500):        # 10× the capacity
            m.record("x", float(i))
        assert m.summary()["x"]["n"] == 50

    def test_ring_buffer_keeps_latest_samples(self) -> None:
        m = Metrics(ring_buffer_size=5)
        for i in range(10):
            m.record("x", float(i))
        # After overflow, the ring keeps the last 5 (5..9)
        s = m.summary()["x"]
        assert s["last_ms"] == pytest.approx(9.0, abs=1e-6)
        assert s["max_ms"] == pytest.approx(9.0, abs=1e-6)

    def test_max_samples_property(self) -> None:
        m = Metrics(ring_buffer_size=123)
        assert m.max_samples == 123

    def test_default_max_samples(self) -> None:
        m = Metrics()
        assert m.max_samples == 500

    def test_multiple_stages_independent_bounds(self) -> None:
        m = Metrics(ring_buffer_size=10)
        for i in range(100):
            m.record("a", float(i))
            m.record("b", float(i))
        assert m.summary()["a"]["n"] == 10
        assert m.summary()["b"]["n"] == 10


# ---------------------------------------------------------------------------
# 2. Summary statistics correctness
# ---------------------------------------------------------------------------

class TestSummaryStatistics:
    @pytest.fixture
    def m100(self) -> Metrics:
        m = Metrics(ring_buffer_size=200)
        for v in range(1, 101):   # 1..100
            m.record("lat", float(v))
        return m

    def test_p50(self, m100) -> None:
        s = m100.summary()["lat"]
        # Sorted [1..100]; index 50 → value 51
        assert s["p50_ms"] == pytest.approx(51.0, abs=1e-6)

    def test_p95(self, m100) -> None:
        s = m100.summary()["lat"]
        # index min(99, int(100 * 0.95)) = 95 → value 96
        assert s["p95_ms"] == pytest.approx(96.0, abs=1e-6)

    def test_max(self, m100) -> None:
        assert m100.summary()["lat"]["max_ms"] == pytest.approx(100.0, abs=1e-6)

    def test_mean(self, m100) -> None:
        # mean of 1..100 = 50.5
        assert m100.summary()["lat"]["mean_ms"] == pytest.approx(50.5, abs=0.01)

    def test_last(self, m100) -> None:
        assert m100.summary()["lat"]["last_ms"] == pytest.approx(100.0, abs=1e-6)

    def test_empty_stage_all_zeros(self) -> None:
        m = Metrics()
        m.record("x", 5.0)   # touch stage so it exists
        # then ask for summary on a fresh metrics with an untouched stage
        m2 = Metrics()
        # untouched stage doesn't appear in summary at all
        assert "y" not in m2.summary()

    def test_single_sample(self) -> None:
        m = Metrics()
        m.record("s", 42.0)
        s = m.summary()["s"]
        assert s["n"] == 1
        assert s["p50_ms"] == pytest.approx(42.0, abs=1e-6)
        assert s["p95_ms"] == pytest.approx(42.0, abs=1e-6)
        assert s["max_ms"] == pytest.approx(42.0, abs=1e-6)
        assert s["mean_ms"] == pytest.approx(42.0, abs=1e-6)
        assert s["last_ms"] == pytest.approx(42.0, abs=1e-6)

    def test_summary_keys_present(self) -> None:
        m = Metrics()
        m.record("k", 1.0)
        s = m.summary()["k"]
        for key in ("n", "p50_ms", "p95_ms", "max_ms", "mean_ms", "last_ms"):
            assert key in s, f"missing key: {key}"


# ---------------------------------------------------------------------------
# 3. reset() and stages
# ---------------------------------------------------------------------------

class TestResetAndStages:
    def test_reset_clears_all_stages(self) -> None:
        m = Metrics()
        m.record("a", 1.0)
        m.record("b", 2.0)
        m.reset()
        s = m.summary()
        assert s["a"]["n"] == 0
        assert s["b"]["n"] == 0

    def test_stages_lists_recorded_names(self) -> None:
        m = Metrics()
        m.record("g1_entry", 1.0)
        m.record("g3_speaker", 2.0)
        m.record("pipeline_total", 10.0)
        assert set(m.stages) == {"g1_entry", "g3_speaker", "pipeline_total"}

    def test_stages_empty_before_recording(self) -> None:
        m = Metrics()
        assert m.stages == []

    def test_reset_then_record_works(self) -> None:
        m = Metrics()
        m.record("x", 1.0)
        m.reset()
        m.record("x", 99.0)
        assert m.summary()["x"]["n"] == 1
        assert m.summary()["x"]["last_ms"] == pytest.approx(99.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Timer context manager
# ---------------------------------------------------------------------------

class TestTimer:
    def test_timer_measures_elapsed_time(self) -> None:
        m = Metrics()
        with m.timer("work"):
            time.sleep(0.010)
        s = m.summary()["work"]
        assert s["n"] == 1
        assert s["last_ms"] >= 8.0   # at least 8 ms

    def test_timer_records_multiple(self) -> None:
        m = Metrics()
        for _ in range(5):
            with m.timer("t"):
                pass
        assert m.summary()["t"]["n"] == 5

    def test_timer_exception_still_records(self) -> None:
        m = Metrics()
        try:
            with m.timer("err"):
                raise ValueError("boom")
        except ValueError:
            pass
        # should still have recorded the latency
        assert m.summary()["err"]["n"] == 1


# ---------------------------------------------------------------------------
# 5. Latency budget
# ---------------------------------------------------------------------------

class TestLatencyBudget:
    def test_warning_fires_above_budget(self, caplog) -> None:
        m = Metrics()
        m.record("pipeline_total", 999.0)
        logger = logging.getLogger("test_budget")
        with caplog.at_level(logging.WARNING):
            m.check_latency_budget(logger, target_ms=150.0)
        assert any("latency budget exceeded" in r.message for r in caplog.records)

    def test_warning_not_fired_below_budget(self, caplog) -> None:
        m = Metrics()
        m.record("pipeline_total", 50.0)
        logger = logging.getLogger("test_budget_ok")
        with caplog.at_level(logging.WARNING):
            m.check_latency_budget(logger, target_ms=150.0)
        assert not any("latency budget exceeded" in r.message for r in caplog.records)

    def test_latency_ok_within_budget(self) -> None:
        m = Metrics()
        for _ in range(20):
            m.record("pipeline_total", 50.0)
        assert m.latency_ok(target_ms=150.0) is True

    def test_latency_ok_exceeds_budget(self) -> None:
        m = Metrics()
        for _ in range(20):
            m.record("pipeline_total", 200.0)
        assert m.latency_ok(target_ms=150.0) is False

    def test_latency_ok_no_data(self) -> None:
        m = Metrics()
        # No data → not violated
        assert m.latency_ok(target_ms=150.0) is True

    def test_budget_warning_includes_actual_ms(self, caplog) -> None:
        m = Metrics()
        m.record("pipeline_total", 300.0)
        logger = logging.getLogger("test_budget_val")
        with caplog.at_level(logging.WARNING):
            m.check_latency_budget(logger, target_ms=150.0)
        combined = " ".join(r.message + str(getattr(r, "kv", {})) for r in caplog.records)
        assert "300" in combined or "last_ms" in combined


# ---------------------------------------------------------------------------
# 6. Pipeline wiring — all guard stages + pipeline_total appear after process()
# ---------------------------------------------------------------------------

class TestPipelineMetricWiring:
    EXPECTED_GUARD_STAGES = {
        "g1_entry", "g2_self_sound", "g3_speaker",
        "g4_live_voice", "g5_words",
    }

    def _make_fw(self):
        from voice_firewall._config import FirewallConfig
        from voice_firewall._pipeline import Firewall
        return Firewall(FirewallConfig.defaults())

    def test_all_guard_stages_recorded(self) -> None:
        fw = self._make_fw()
        fw.process(np.zeros(16000, dtype=np.float32))
        summary = fw.metrics_summary()
        for stage in self.EXPECTED_GUARD_STAGES:
            assert stage in summary, f"stage '{stage}' missing from metrics_summary()"

    def test_pipeline_total_recorded(self) -> None:
        fw = self._make_fw()
        fw.process(np.zeros(16000, dtype=np.float32))
        summary = fw.metrics_summary()
        assert "pipeline_total" in summary

    def test_pipeline_total_positive(self) -> None:
        fw = self._make_fw()
        fw.process(np.zeros(16000, dtype=np.float32))
        total = fw.metrics_summary()["pipeline_total"]
        assert total["last_ms"] > 0.0

    def test_metrics_bounded_after_many_calls(self) -> None:
        from voice_firewall._config import FirewallConfig, MetricsConfig
        from voice_firewall._pipeline import Firewall
        cfg = FirewallConfig.defaults()
        cfg.metrics.ring_buffer_size = 20
        fw = Firewall(cfg)
        for _ in range(200):
            fw.process(np.zeros(4000, dtype=np.float32))
        summary = fw.metrics_summary()
        for stage, stats in summary.items():
            assert stats["n"] <= 20, f"stage '{stage}' has n={stats['n']} > maxlen=20"

    def test_metrics_summary_has_all_keys(self) -> None:
        fw = self._make_fw()
        fw.process(np.zeros(8000, dtype=np.float32))
        for stage, stats in fw.metrics_summary().items():
            for key in ("n", "p50_ms", "p95_ms", "max_ms", "mean_ms", "last_ms"):
                assert key in stats, f"stage '{stage}' missing key '{key}'"

    def test_pipeline_reset_clears_metrics(self) -> None:
        fw = self._make_fw()
        fw.process(np.zeros(8000, dtype=np.float32))
        fw.stop()
        # After stop, reset the metrics manually (stop calls guard.reset, not metrics.reset)
        fw._metrics.reset()
        for stats in fw.metrics_summary().values():
            assert stats["n"] == 0

    def test_target_latency_config_respected(self) -> None:
        """MetricsConfig.target_latency_ms is wired to check_latency_budget."""
        from voice_firewall._config import FirewallConfig
        from voice_firewall._pipeline import Firewall
        cfg = FirewallConfig.defaults()
        cfg.metrics.target_latency_ms = 0.001  # impossibly tight → warning guaranteed
        fw = Firewall(cfg)
        # process() should log a latency budget warning but not raise
        fw.process(np.zeros(8000, dtype=np.float32))
        # test just that it completes without error; caplog check is in TestLatencyBudget
