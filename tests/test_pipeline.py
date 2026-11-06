"""End-to-end pipeline tests (no hardware, no models, no network)."""
from __future__ import annotations

import numpy as np
import pytest

from voice_firewall import Firewall, FirewallConfig, TaggedSegment
from voice_firewall._types import ActionVerdict, GuardVerdict, RawSegment


@pytest.fixture
def fw() -> Firewall:
    fw = Firewall(FirewallConfig.defaults())
    fw.start()
    yield fw
    fw.stop()


class TestFirewallLifecycle:
    def test_start_stop(self) -> None:
        fw = Firewall(FirewallConfig.defaults())
        fw.start()
        fw.stop()

    def test_double_stop_safe(self) -> None:
        fw = Firewall(FirewallConfig.defaults())
        fw.start()
        fw.stop()
        fw.stop()  # second stop must not raise


class TestProcessSegment:
    def test_passes_audio_returns_tagged(self, fw, one_second_audio) -> None:
        result = fw.process(one_second_audio)
        assert isinstance(result, TaggedSegment)

    def test_tagged_has_audio(self, fw, one_second_audio) -> None:
        result = fw.process(one_second_audio)
        assert result is not None
        assert result.audio.shape[0] == one_second_audio.shape[0]

    def test_guard_trace_has_five_entries(self, fw, one_second_audio) -> None:
        result = fw.process(one_second_audio)
        assert result is not None
        assert len(result.guard_trace) == 5

    def test_all_guards_pass_in_stub_mode(self, fw, one_second_audio) -> None:
        result = fw.process(one_second_audio)
        assert result is not None
        for gr in result.guard_trace:
            assert gr.verdict == GuardVerdict.PASS

    def test_process_returns_none_when_guard_drops(self, fw) -> None:
        from unittest.mock import patch
        from voice_firewall._types import GuardResult
        drop = GuardResult(guard="g1_entry", verdict=GuardVerdict.DROP, reason="test drop")
        audio = np.zeros(320, dtype=np.float32)
        with patch.object(fw._g1, "_check", return_value=drop):
            result = fw.process(audio)
        assert result is None

    def test_latency_recorded_in_metrics(self, fw, one_second_audio) -> None:
        fw.process(one_second_audio)
        summary = fw.metrics_summary()
        assert "pipeline_total" in summary
        assert summary["pipeline_total"]["n"] == 1


class TestAuthorize:
    def test_stub_allows_all_actions(self, fw, one_second_audio) -> None:
        tagged = fw.process(one_second_audio)
        decision = fw.authorize("open_app", {"app": "calc"}, tagged)
        assert decision.verdict == ActionVerdict.ALLOW

    def test_authorize_without_segment(self, fw) -> None:
        decision = fw.authorize("read_text", {}, None)
        assert decision.verdict == ActionVerdict.ALLOW

    def test_audit_id_unique_per_call(self, fw) -> None:
        d1 = fw.authorize("open_app", {}, None)
        d2 = fw.authorize("open_app", {}, None)
        assert d1.audit_id != d2.audit_id


class TestContextControl:
    def test_set_assistant_speaking(self, fw) -> None:
        fw.set_assistant_speaking(True)
        assert fw._ctx.assistant_speaking is True
        fw.set_assistant_speaking(False)
        assert fw._ctx.assistant_speaking is False

    def test_mark_untrusted_content(self, fw) -> None:
        assert fw._ctx.untrusted_content_seen is False
        fw.mark_untrusted_content()
        assert fw._ctx.untrusted_content_seen is True

    def test_clear_untrusted_flag(self, fw) -> None:
        fw.mark_untrusted_content()
        fw.clear_untrusted_flag()
        assert fw._ctx.untrusted_content_seen is False

    def test_set_reference_audio_does_not_raise(self, fw) -> None:
        frame = np.zeros(320, dtype=np.float32)
        fw.set_reference_audio(frame)  # stub — must not raise
