"""Tests for Guard 1 — Entry guard (stub phase)."""
from __future__ import annotations

from voice_firewall._types import GuardVerdict
from voice_firewall.guards.g1_entry import EntryGuard


class TestEntryGuardStub:
    def test_passes_audio_segment(self, speech_segment, ctx) -> None:
        g = EntryGuard()
        result = g.check(speech_segment, ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_guard_name(self) -> None:
        assert EntryGuard().name == "g1_entry"

    def test_fail_closed_on_exception(self, ctx) -> None:
        """GuardBase must catch exceptions and return DROP."""
        from unittest.mock import patch
        from voice_firewall._types import RawSegment
        import numpy as np

        g = EntryGuard()
        seg = RawSegment(audio=np.zeros(320, dtype=np.float32))

        with patch.object(g, "_check", side_effect=RuntimeError("boom")):
            result = g.check(seg, ctx)

        assert result.verdict == GuardVerdict.DROP
        assert "boom" in result.reason

    def test_latency_recorded(self, speech_segment, ctx) -> None:
        g = EntryGuard()
        result = g.check(speech_segment, ctx)
        assert result.latency_ms >= 0.0
