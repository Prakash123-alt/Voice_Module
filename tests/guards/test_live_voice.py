"""Tests for Guard 4 — Live-voice guard (stub phase)."""
from voice_firewall._config import LiveVoiceConfig
from voice_firewall._types import GuardVerdict
from voice_firewall.guards.g4_live_voice import LiveVoiceGuard


class TestLiveVoiceGuardStub:
    def test_passes_when_disabled(self, speech_segment, ctx) -> None:
        cfg = LiveVoiceConfig(enabled=False)
        result = LiveVoiceGuard(cfg).check(speech_segment, ctx)
        assert result.verdict == GuardVerdict.PASS
        assert result.tags.get("liveness_score") is None

    def test_passes_when_enabled_stub(self, speech_segment, ctx) -> None:
        cfg = LiveVoiceConfig(enabled=True)
        result = LiveVoiceGuard(cfg).check(speech_segment, ctx)
        # Stub always passes; model not loaded yet
        assert result.verdict == GuardVerdict.PASS

    def test_guard_name(self) -> None:
        assert LiveVoiceGuard(LiveVoiceConfig()).name == "g4_live_voice"

    def test_fail_closed(self, ctx) -> None:
        from unittest.mock import patch
        from voice_firewall._types import RawSegment
        import numpy as np
        g = LiveVoiceGuard(LiveVoiceConfig())
        seg = RawSegment(audio=np.zeros(320, dtype=np.float32))
        with patch.object(g, "_check", side_effect=RuntimeError("model error")):
            result = g.check(seg, ctx)
        assert result.verdict == GuardVerdict.DROP
