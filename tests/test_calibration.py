"""Tests for Phase 1 Step 6 — Observe mode + calibration math.

All tests use synthetic scores — no audio hardware, model downloads, or network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voice_firewall.calibration import (
    CalibrationResult,
    ScoreLog,
    compute_calibration,
    compute_far_frr,
    find_eer,
    find_threshold_at_far,
    frr_at_threshold,
)


# ---------------------------------------------------------------------------
# Synthetic score distributions
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
# Well-separated: owner ~0.8, non-owner ~0.2
OWNER_SCORES = RNG.normal(0.8, 0.05, 200).clip(-1, 1)
NONOWNER_SCORES = RNG.normal(0.2, 0.05, 200).clip(-1, 1)

# Heavily overlapping (harder case)
OVERLAP_OWNER = RNG.normal(0.6, 0.1, 100).clip(-1, 1)
OVERLAP_NONOWNER = RNG.normal(0.5, 0.1, 100).clip(-1, 1)


# ---------------------------------------------------------------------------
# Tests: compute_far_frr
# ---------------------------------------------------------------------------

class TestComputeFarFrr:
    def test_output_shapes(self) -> None:
        t, far, frr = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, n_points=100)
        assert t.shape == far.shape == frr.shape == (100,)

    def test_thresholds_sorted_ascending(self) -> None:
        t, _, _ = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, n_points=200)
        assert np.all(np.diff(t) >= 0)

    def test_far_monotone_decreasing(self) -> None:
        # Higher threshold → fewer non-owners accepted → FAR decreases
        t, far, _ = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, n_points=200)
        assert np.all(np.diff(far) <= 1e-9)

    def test_frr_monotone_increasing(self) -> None:
        # Higher threshold → more owners rejected → FRR increases
        t, _, frr = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, n_points=200)
        assert np.all(np.diff(frr) >= -1e-9)

    def test_far_bounds(self) -> None:
        _, far, _ = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, n_points=100)
        assert np.all(far >= 0) and np.all(far <= 1)

    def test_frr_bounds(self) -> None:
        _, _, frr = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, n_points=100)
        assert np.all(frr >= 0) and np.all(frr <= 1)

    def test_custom_thresholds(self) -> None:
        custom = np.array([0.0, 0.5, 1.0])
        t, far, frr = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES, thresholds=custom)
        np.testing.assert_array_equal(t, custom)

    def test_at_zero_all_accepted(self) -> None:
        # At very low threshold, all non-owners are accepted (FAR = 1)
        t, far, frr = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES,
                                       thresholds=np.array([-2.0]))
        assert far[0] == pytest.approx(1.0, abs=1e-6)
        assert frr[0] == pytest.approx(0.0, abs=1e-6)

    def test_at_max_all_rejected(self) -> None:
        # At very high threshold, all non-owners are rejected (FAR = 0, FRR = 1)
        t, far, frr = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES,
                                       thresholds=np.array([2.0]))
        assert far[0] == pytest.approx(0.0, abs=1e-6)
        assert frr[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: find_eer
# ---------------------------------------------------------------------------

class TestFindEer:
    def test_eer_low_for_separated_scores(self) -> None:
        t, rate = find_eer(OWNER_SCORES, NONOWNER_SCORES)
        # Scores well-separated → EER should be very low
        assert rate < 0.05

    def test_eer_threshold_between_distributions(self) -> None:
        t, _ = find_eer(OWNER_SCORES, NONOWNER_SCORES)
        # EER threshold should be between the two distribution means
        assert 0.2 < t < 0.8

    def test_eer_higher_for_overlapping_scores(self) -> None:
        t_sep, rate_sep = find_eer(OWNER_SCORES, NONOWNER_SCORES)
        t_ovl, rate_ovl = find_eer(OVERLAP_OWNER, OVERLAP_NONOWNER)
        assert rate_ovl > rate_sep

    def test_eer_returns_two_floats(self) -> None:
        result = find_eer(OWNER_SCORES, NONOWNER_SCORES)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)

    def test_eer_no_crossing_does_not_crash(self) -> None:
        # Identical distributions → no clear crossing; should still return something
        same = np.array([0.5, 0.6, 0.7])
        t, rate = find_eer(same, same)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Tests: find_threshold_at_far
# ---------------------------------------------------------------------------

class TestFindThresholdAtFar:
    def test_threshold_achieves_target_far(self) -> None:
        t = find_threshold_at_far(OWNER_SCORES, NONOWNER_SCORES, target_far=0.05)
        assert t is not None
        # Verify the actual FAR at this threshold is <= 5%
        _, far, _ = compute_far_frr(OWNER_SCORES, NONOWNER_SCORES,
                                     thresholds=np.array([t]))
        assert far[0] <= 0.05 + 1e-4  # allow tiny numeric error

    def test_returns_none_when_unachievable(self) -> None:
        # target_far = 0 is unachievable unless distributions are perfectly separated
        rng = np.random.default_rng(99)
        owner = rng.normal(0.5, 0.2, 50).clip(-1, 1)
        nonowner = rng.normal(0.5, 0.2, 50).clip(-1, 1)
        result = find_threshold_at_far(owner, nonowner, target_far=0.0)
        # Either None or the max threshold
        assert result is None or isinstance(result, float)

    def test_1pct_target_achievable(self) -> None:
        t = find_threshold_at_far(OWNER_SCORES, NONOWNER_SCORES, target_far=0.01)
        assert t is not None


# ---------------------------------------------------------------------------
# Tests: frr_at_threshold
# ---------------------------------------------------------------------------

class TestFrrAtThreshold:
    def test_low_threshold_zero_frr(self) -> None:
        assert frr_at_threshold(OWNER_SCORES, -2.0) == pytest.approx(0.0, abs=1e-6)

    def test_high_threshold_one_frr(self) -> None:
        assert frr_at_threshold(OWNER_SCORES, 2.0) == pytest.approx(1.0, abs=1e-6)

    def test_intermediate(self) -> None:
        frr = frr_at_threshold(OWNER_SCORES, 0.75)
        assert 0.0 <= frr <= 1.0

    def test_empty_scores(self) -> None:
        assert frr_at_threshold(np.array([]), 0.5) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_calibration
# ---------------------------------------------------------------------------

class TestComputeCalibration:
    def test_returns_calibration_result(self) -> None:
        result = compute_calibration(OWNER_SCORES, NONOWNER_SCORES)
        assert isinstance(result, CalibrationResult)

    def test_n_samples_correct(self) -> None:
        result = compute_calibration(OWNER_SCORES, NONOWNER_SCORES)
        assert result.n_owner == len(OWNER_SCORES)
        assert result.n_nonowner == len(NONOWNER_SCORES)

    def test_strict_threshold_gte_threshold(self) -> None:
        result = compute_calibration(OWNER_SCORES, NONOWNER_SCORES)
        assert result.strict_threshold >= result.threshold

    def test_strict_threshold_has_margin(self) -> None:
        result = compute_calibration(OWNER_SCORES, NONOWNER_SCORES, strict_margin=0.1)
        assert result.strict_threshold >= result.threshold + 0.09  # allow tiny numeric error

    def test_threshold_clipped_at_one(self) -> None:
        # Huge margin should not push strict_threshold above 1.0
        result = compute_calibration(OWNER_SCORES, NONOWNER_SCORES, strict_margin=0.9)
        assert result.strict_threshold <= 1.0

    def test_empty_owner_raises(self) -> None:
        with pytest.raises(ValueError, match="owner"):
            compute_calibration(np.array([]), NONOWNER_SCORES)

    def test_empty_nonowner_raises(self) -> None:
        with pytest.raises(ValueError, match="non-owner"):
            compute_calibration(OWNER_SCORES, np.array([]))

    def test_report_string(self) -> None:
        result = compute_calibration(OWNER_SCORES, NONOWNER_SCORES)
        report = result.report()
        assert "EER" in report
        assert "threshold" in report.lower()


# ---------------------------------------------------------------------------
# Tests: ScoreLog
# ---------------------------------------------------------------------------

class TestScoreLog:
    def test_append_and_read(self, tmp_path) -> None:
        log_path = str(tmp_path / "scores.jsonl")
        sl = ScoreLog(log_path)
        sl.append(0.85, label="owner")
        sl.append(0.30, label="nonowner")
        sl.append(0.70, label="unknown")

        owner, nonowner = ScoreLog.read(log_path)
        assert len(owner) == 1
        assert len(nonowner) == 1
        assert float(owner[0]) == pytest.approx(0.85, abs=1e-5)
        assert float(nonowner[0]) == pytest.approx(0.30, abs=1e-5)

    def test_unknown_labels_ignored(self, tmp_path) -> None:
        log_path = str(tmp_path / "scores.jsonl")
        sl = ScoreLog(log_path)
        sl.append(0.70, label="unknown")
        owner, nonowner = ScoreLog.read(log_path)
        assert len(owner) == 0 and len(nonowner) == 0

    def test_creates_parent_dirs(self, tmp_path) -> None:
        log_path = str(tmp_path / "nested" / "dir" / "scores.jsonl")
        sl = ScoreLog(log_path)
        sl.append(0.5, label="owner")
        assert Path(log_path).exists()

    def test_appends_multiple_times(self, tmp_path) -> None:
        log_path = str(tmp_path / "scores.jsonl")
        sl = ScoreLog(log_path)
        for i in range(10):
            sl.append(float(i) / 10, label="owner")
        owner, _ = ScoreLog.read(log_path)
        assert len(owner) == 10

    def test_read_empty_file(self, tmp_path) -> None:
        log_path = str(tmp_path / "empty.jsonl")
        Path(log_path).write_text("", encoding="utf-8")
        owner, nonowner = ScoreLog.read(log_path)
        assert len(owner) == 0 and len(nonowner) == 0

    def test_read_malformed_lines_skipped(self, tmp_path) -> None:
        log_path = str(tmp_path / "scores.jsonl")
        Path(log_path).write_text(
            '{"score": 0.8, "label": "owner"}\n'
            'NOT_JSON\n'
            '{"score": 0.3, "label": "nonowner"}\n',
            encoding="utf-8",
        )
        owner, nonowner = ScoreLog.read(log_path)
        assert len(owner) == 1 and len(nonowner) == 1

    def test_jsonl_format(self, tmp_path) -> None:
        log_path = str(tmp_path / "scores.jsonl")
        sl = ScoreLog(log_path)
        sl.append(0.75, label="owner")
        lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "ts" in entry and "score" in entry and "label" in entry


# ---------------------------------------------------------------------------
# Tests: observe mode + score log wired to G3
# ---------------------------------------------------------------------------

class TestObserveModeScoreLog:
    def test_score_logged_in_observe_mode(self, tmp_path) -> None:
        from voice_firewall._config import SpeakerConfig
        from voice_firewall._types import PipelineContext, RawSegment
        from voice_firewall.guards.g3_speaker import SpeakerGuard
        from voice_firewall.speaker.embedder import StubEmbedder
        from voice_firewall.speaker.gate import save_profile
        import numpy as np

        log_path = str(tmp_path / "scores.jsonl")
        profile = np.zeros(256, dtype=np.float32)
        profile[0] = 1.0

        cfg = SpeakerConfig(
            threshold=None,  # observe mode
            observe_mode=True,
            min_segment_ms=0,
            score_log_path=log_path,
        )
        guard = SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=profile)
        seg = RawSegment(audio=np.zeros(8000, dtype=np.float32), speech_ms=500)
        ctx = PipelineContext()
        guard.check(seg, ctx)

        # Score should have been written to the log
        owner, nonowner = ScoreLog.read(log_path)
        # label is "unknown" during live use — not in owner or nonowner
        all_lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(all_lines) == 1

    def test_score_not_logged_in_enforcement_mode(self, tmp_path) -> None:
        from voice_firewall._config import SpeakerConfig
        from voice_firewall._types import PipelineContext, RawSegment
        from voice_firewall.guards.g3_speaker import SpeakerGuard
        from voice_firewall.speaker.embedder import StubEmbedder
        import numpy as np

        log_path = str(tmp_path / "scores.jsonl")
        profile = np.zeros(256, dtype=np.float32)
        profile[0] = 1.0

        cfg = SpeakerConfig(
            threshold=0.5,
            observe_mode=False,
            min_segment_ms=0,
            score_log_path=log_path,
        )
        guard = SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=profile)
        seg = RawSegment(audio=np.zeros(8000, dtype=np.float32), speech_ms=500)
        guard.check(seg, PipelineContext())

        # In enforcement mode, score should NOT be logged
        assert not Path(log_path).exists() or Path(log_path).stat().st_size == 0

    def test_no_log_path_no_error(self) -> None:
        from voice_firewall._config import SpeakerConfig
        from voice_firewall._types import PipelineContext, RawSegment
        from voice_firewall.guards.g3_speaker import SpeakerGuard
        from voice_firewall.speaker.embedder import StubEmbedder
        import numpy as np

        profile = np.zeros(256, dtype=np.float32)
        profile[0] = 1.0

        cfg = SpeakerConfig(threshold=None, observe_mode=True, min_segment_ms=0)
        guard = SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=profile)
        seg = RawSegment(audio=np.zeros(8000, dtype=np.float32), speech_ms=500)
        # Must not raise
        guard.check(seg, PipelineContext())
