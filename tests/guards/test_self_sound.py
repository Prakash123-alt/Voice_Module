"""Tests for Guard 2 — Self-sound guard (full implementation)."""
from __future__ import annotations

import numpy as np
import pytest

from voice_firewall._config import SelfSoundConfig
from voice_firewall._types import GuardVerdict, RawSegment
from voice_firewall.guards.g2_self_sound import SelfSoundGuard


def _seg(suppression_db=None, source="mic") -> RawSegment:
    audio = np.zeros(320, dtype=np.float32)
    return RawSegment(audio=audio, source=source, aec_suppression_db=suppression_db)


@pytest.fixture
def cfg() -> SelfSoundConfig:
    return SelfSoundConfig(loopback_suspect_db=12.0)


@pytest.fixture
def guard(cfg) -> SelfSoundGuard:
    return SelfSoundGuard(cfg)


class TestLoopbackUnavailable:
    def test_passes_when_no_suppression_data(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=None), ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_loopback_available_tag_false(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=None), ctx)
        assert result.tags["loopback_available"] is False

    def test_source_unchanged_when_unavailable(self, guard, ctx) -> None:
        seg = _seg(suppression_db=None, source="mic")
        guard.check(seg, ctx)
        assert seg.source == "mic"


class TestMicAudio:
    def test_low_suppression_tagged_mic(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=2.0), ctx)
        assert result.tags["source"] == "mic"

    def test_verdict_pass_for_mic(self, guard, ctx) -> None:
        assert guard.check(_seg(suppression_db=0.0), ctx).verdict == GuardVerdict.PASS

    def test_zero_suppression_is_mic(self, guard, ctx) -> None:
        seg = _seg(suppression_db=0.0)
        guard.check(seg, ctx)
        assert seg.source == "mic"

    def test_just_below_threshold_is_mic(self, guard, ctx) -> None:
        seg = _seg(suppression_db=11.99)
        guard.check(seg, ctx)
        assert seg.source == "mic"

    def test_negative_suppression_is_mic(self, guard, ctx) -> None:
        # AEC can amplify slightly in edge cases; still mic
        result = guard.check(_seg(suppression_db=-3.0), ctx)
        assert result.tags["source"] == "mic"


class TestLoopbackSuspect:
    def test_at_exact_threshold_is_suspect(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=12.0), ctx)
        assert result.tags["source"] == "loopback-suspect"

    def test_above_threshold_is_suspect(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=20.0), ctx)
        assert result.tags["source"] == "loopback-suspect"

    def test_segment_source_updated_in_place(self, guard, ctx) -> None:
        seg = _seg(suppression_db=20.0, source="mic")
        guard.check(seg, ctx)
        assert seg.source == "loopback-suspect"

    def test_still_passes_not_drops(self, guard, ctx) -> None:
        # G2 annotates; never drops
        result = guard.check(_seg(suppression_db=99.0), ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_loopback_available_tag_true(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=20.0), ctx)
        assert result.tags["loopback_available"] is True

    def test_suppression_db_in_tags(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=15.5), ctx)
        assert result.tags["aec_suppression_db"] == pytest.approx(15.5, abs=0.01)


class TestCustomThreshold:
    def test_custom_threshold_respected(self, ctx) -> None:
        g = SelfSoundGuard(SelfSoundConfig(loopback_suspect_db=6.0))
        assert g.check(_seg(suppression_db=6.0), ctx).tags["source"] == "loopback-suspect"
        assert g.check(_seg(suppression_db=5.99), ctx).tags["source"] == "mic"

    def test_very_high_threshold_accepts_normally(self, ctx) -> None:
        g = SelfSoundGuard(SelfSoundConfig(loopback_suspect_db=50.0))
        assert g.check(_seg(suppression_db=20.0), ctx).tags["source"] == "mic"


class TestGuardContract:
    def test_guard_name(self, guard) -> None:
        assert guard.name == "g2_self_sound"

    def test_fail_closed_on_exception(self, guard, ctx) -> None:
        from unittest.mock import patch
        seg = _seg(suppression_db=5.0)
        with patch.object(guard, "_check", side_effect=RuntimeError("aec crash")):
            result = guard.check(seg, ctx)
        assert result.verdict == GuardVerdict.DROP
        assert "aec crash" in result.reason

    def test_latency_recorded(self, guard, ctx) -> None:
        result = guard.check(_seg(suppression_db=5.0), ctx)
        assert result.latency_ms >= 0.0
