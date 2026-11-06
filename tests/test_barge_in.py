"""Tests for Phase 1 Step 5 — Stricter-while-speaking rule.

Verifies:
1. strict_threshold >= threshold enforced at config load (auto-clamp + warning)
2. Same audio PASS normal, DROP strict (duck-verify-confirm flow)
3. TaggedSegment.interrupt=True only when speaker is verified during barge-in
4. on_barge_in callback fires before guards when assistant is speaking
5. TaggedSegment.interrupt=False for normal (non-barge-in) passes
"""
from __future__ import annotations

import numpy as np
import pytest

from voice_firewall._config import FirewallConfig, SpeakerConfig
from voice_firewall._pipeline import Firewall
from voice_firewall._types import GuardVerdict, PipelineContext, RawSegment
from voice_firewall.guards.g3_speaker import SpeakerGuard
from voice_firewall.speaker.embedder import SpeakerEmbedder


# ---------------------------------------------------------------------------
# Custom embedder that returns a controllable fixed vector
# ---------------------------------------------------------------------------

class FixedEmbedder(SpeakerEmbedder):
    """Returns the fixed vector set via .set_vector()."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0
        self._vec = v

    def set_vector(self, v: np.ndarray) -> None:
        self._vec = v.astype(np.float32)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        return self._vec.copy()

    @property
    def dim(self) -> int:
        return self._dim


def _profile_at_index(idx: int, dim: int = 256) -> np.ndarray:
    """Unit vector with 1.0 at index idx."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _embedding_with_score(target_score: float, dim: int = 256) -> np.ndarray:
    """Return an L2-normalised embedding with cosine similarity `target_score`
    against the unit vector [1, 0, 0, ...].
    
    Uses the identity: v = [score, sqrt(1 - score²), 0, ...].
    """
    score = float(np.clip(target_score, -1.0, 1.0))
    complement = float(np.sqrt(max(0.0, 1.0 - score ** 2)))
    v = np.zeros(dim, dtype=np.float32)
    v[0] = score
    if dim > 1:
        v[1] = complement
    return v  # already unit length


PROFILE = _profile_at_index(0)   # owner profile: unit vector at index 0
AUDIO   = np.zeros(8000, dtype=np.float32)  # 500 ms of silence


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestStrictThresholdValidation:
    def test_valid_config_unchanged(self) -> None:
        cfg = SpeakerConfig(threshold=0.5, strict_threshold=0.8)
        assert cfg.strict_threshold == pytest.approx(0.8, abs=1e-6)

    def test_equal_thresholds_ok(self) -> None:
        cfg = SpeakerConfig(threshold=0.6, strict_threshold=0.6)
        assert cfg.strict_threshold == pytest.approx(0.6, abs=1e-6)

    def test_looser_strict_clamped_up(self) -> None:
        cfg = SpeakerConfig(threshold=0.7, strict_threshold=0.3)
        # strict_threshold must be clamped to threshold
        assert cfg.strict_threshold == pytest.approx(0.7, abs=1e-6)

    def test_looser_strict_logged(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.WARNING):
            SpeakerConfig(threshold=0.7, strict_threshold=0.3)
        assert any("strict_threshold" in r.message.lower() or
                   "clamping" in r.message.lower()
                   for r in caplog.records)

    def test_none_thresholds_no_error(self) -> None:
        cfg = SpeakerConfig(threshold=None, strict_threshold=None)
        assert cfg.threshold is None and cfg.strict_threshold is None

    def test_only_threshold_set_no_error(self) -> None:
        cfg = SpeakerConfig(threshold=0.5, strict_threshold=None)
        assert cfg.strict_threshold is None


# ---------------------------------------------------------------------------
# 2. Same embedding: PASS normal, DROP strict
# ---------------------------------------------------------------------------

class TestStrictGateDifferentiation:
    """Score = 0.6 → passes normal (0.5) but drops strict (0.8)."""

    @pytest.fixture
    def embedder(self) -> FixedEmbedder:
        e = FixedEmbedder()
        e.set_vector(_embedding_with_score(0.6))
        return e

    @pytest.fixture
    def guard(self, embedder) -> SpeakerGuard:
        cfg = SpeakerConfig(
            threshold=0.5,
            strict_threshold=0.8,
            observe_mode=False,
            min_segment_ms=100,
        )
        return SpeakerGuard(cfg=cfg, embedder=embedder, profile=PROFILE)

    def test_pass_when_not_speaking(self, guard) -> None:
        ctx = PipelineContext(assistant_speaking=False)
        seg = RawSegment(audio=AUDIO, speech_ms=500)
        result = guard.check(seg, ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_drop_when_assistant_speaking(self, guard) -> None:
        ctx = PipelineContext(assistant_speaking=True)
        seg = RawSegment(audio=AUDIO, speech_ms=500)
        result = guard.check(seg, ctx)
        assert result.verdict == GuardVerdict.DROP

    def test_threshold_used_normal(self, guard) -> None:
        ctx = PipelineContext(assistant_speaking=False)
        seg = RawSegment(audio=AUDIO, speech_ms=500)
        result = guard.check(seg, ctx)
        assert result.tags["threshold_used"] == pytest.approx(0.5, abs=1e-6)

    def test_threshold_used_strict(self, guard) -> None:
        ctx = PipelineContext(assistant_speaking=True)
        seg = RawSegment(audio=AUDIO, speech_ms=500)
        result = guard.check(seg, ctx)
        assert result.tags["threshold_used"] == pytest.approx(0.8, abs=1e-6)

    def test_high_score_passes_both(self, embedder) -> None:
        embedder.set_vector(_embedding_with_score(0.95))
        cfg = SpeakerConfig(threshold=0.5, strict_threshold=0.8,
                             observe_mode=False, min_segment_ms=100)
        guard = SpeakerGuard(cfg=cfg, embedder=embedder, profile=PROFILE)
        for speaking in (False, True):
            ctx = PipelineContext(assistant_speaking=speaking)
            seg = RawSegment(audio=AUDIO, speech_ms=500)
            assert guard.check(seg, ctx).verdict == GuardVerdict.PASS


# ---------------------------------------------------------------------------
# 3. interrupt flag
# ---------------------------------------------------------------------------

def _make_fw_with_stub(
    tmp_path,
    profile: np.ndarray,
    threshold: float = 0.5,
    strict_threshold: float = 0.5,
    observe_mode: bool = False,
) -> Firewall:
    """Build a Firewall whose G3 uses StubEmbedder and a known profile.

    Avoids ONNX model download so the test is fully deterministic.
    StubEmbedder always returns the unit vector at index 0, so:
      cosine(embed_out, profile_at_0) = 1.0  (owner profile)
      cosine(embed_out, profile_at_1) = 0.0  (orthogonal profile)
    """
    cfg = FirewallConfig.defaults()
    cfg.guards.speaker.threshold = threshold
    cfg.guards.speaker.strict_threshold = strict_threshold
    cfg.guards.speaker.observe_mode = observe_mode
    cfg.guards.speaker.min_segment_ms = 0

    profile_path = str(tmp_path / "owner.npy")
    np.save(profile_path, profile)
    cfg.guards.speaker.profile_path = profile_path

    fw = Firewall(cfg)
    # Directly replace G3's embedder so no ONNX model is needed in tests.
    from voice_firewall.guards.g3_speaker import SpeakerGuard
    from voice_firewall.speaker.embedder import StubEmbedder as _Stub
    fw._g3 = SpeakerGuard(
        cfg=cfg.guards.speaker,
        embedder=_Stub(),
        profile=profile,
    )
    fw._guards[2] = fw._g3  # keep the guard list consistent
    return fw


class TestInterruptFlag:
    @pytest.fixture
    def fw(self, tmp_path) -> Firewall:
        # StubEmbedder returns v[0]=1.0; profile is also v[0]=1.0 → score 1.0 → passes
        profile = _profile_at_index(0)
        return _make_fw_with_stub(tmp_path, profile, threshold=0.5, strict_threshold=0.5)

    def test_interrupt_true_during_barge_in(self, fw) -> None:
        fw.set_assistant_speaking(True)
        audio = np.zeros(16000, dtype=np.float32)
        tagged = fw.process(audio)
        assert tagged is not None
        assert tagged.interrupt is True

    def test_interrupt_false_normal_turn(self, fw) -> None:
        fw.set_assistant_speaking(False)
        audio = np.zeros(16000, dtype=np.float32)
        tagged = fw.process(audio)
        assert tagged is not None
        assert tagged.interrupt is False

    def test_interrupt_false_on_drop(self, tmp_path) -> None:
        """Dropped segments never produce a TaggedSegment."""
        # Orthogonal profile → score = 0.0 < 0.5 → G3 DROPs
        profile = _profile_at_index(1)  # orthogonal to StubEmbedder output
        fw = _make_fw_with_stub(tmp_path, profile, threshold=0.5, strict_threshold=0.5)
        fw.set_assistant_speaking(True)
        result = fw.process(np.zeros(16000, dtype=np.float32))
        assert result is None  # dropped — no TaggedSegment


# ---------------------------------------------------------------------------
# 4. on_barge_in callback
# ---------------------------------------------------------------------------

class TestBargeInCallback:
    def _make_fw(self, callback) -> Firewall:
        cfg = FirewallConfig.defaults()
        return Firewall(cfg, on_barge_in=callback)

    def test_callback_fires_when_assistant_speaking(self) -> None:
        fired = []
        fw = self._make_fw(lambda: fired.append(True))
        fw.set_assistant_speaking(True)
        fw.process(np.zeros(8000, dtype=np.float32))
        assert len(fired) == 1

    def test_callback_not_fired_when_not_speaking(self) -> None:
        fired = []
        fw = self._make_fw(lambda: fired.append(True))
        fw.set_assistant_speaking(False)
        fw.process(np.zeros(8000, dtype=np.float32))
        assert len(fired) == 0

    def test_callback_exception_does_not_crash_pipeline(self) -> None:
        def bad_cb():
            raise RuntimeError("TTS driver error")
        fw = self._make_fw(bad_cb)
        fw.set_assistant_speaking(True)
        # Must not raise
        fw.process(np.zeros(8000, dtype=np.float32))

    def test_no_callback_does_not_raise(self) -> None:
        fw = Firewall(FirewallConfig.defaults())
        fw.set_assistant_speaking(True)
        fw.process(np.zeros(8000, dtype=np.float32))
