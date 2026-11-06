"""Tests for Guard 3 — Speaker guard (Phase 1 Step 4).

All tests use StubEmbedder — no ONNX model, no network required.
"""
from __future__ import annotations

import numpy as np
import pytest

from voice_firewall._config import SpeakerConfig
from voice_firewall._types import GuardVerdict, PipelineContext, RawSegment
from voice_firewall.guards.g3_speaker import SpeakerGuard
from voice_firewall.speaker.embedder import StubEmbedder
from voice_firewall.speaker.gate import (
    CosineSpeakerGate,
    GateResult,
    RollingEmbeddingGate,
    load_profile,
    save_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(speech_ms=500) -> RawSegment:
    n = int(speech_ms * 16)  # 16000 Hz × (ms/1000)
    return RawSegment(audio=np.random.randn(n).astype(np.float32), speech_ms=speech_ms)


def _stub_profile() -> np.ndarray:
    """Unit vector that the StubEmbedder will score 1.0 against."""
    v = np.zeros(256, dtype=np.float32)
    v[0] = 1.0
    return v


def _anti_profile() -> np.ndarray:
    """Profile orthogonal to the StubEmbedder output → score 0.0."""
    v = np.zeros(256, dtype=np.float32)
    v[1] = 1.0
    return v


def _guard(
    profile=None,
    threshold: float = 0.5,
    strict_threshold: float = 0.7,
    observe_mode: bool = False,
) -> SpeakerGuard:
    cfg = SpeakerConfig(
        threshold=threshold,
        strict_threshold=strict_threshold,
        observe_mode=observe_mode,
        min_segment_ms=100,
    )
    return SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=profile)


# ---------------------------------------------------------------------------
# Tests: no profile enrolled
# ---------------------------------------------------------------------------

class TestNoProfile:
    def test_passes_without_profile(self, ctx) -> None:
        g = _guard(profile=None)
        result = g.check(_seg(), ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_enrolled_tag_false(self, ctx) -> None:
        g = _guard(profile=None)
        result = g.check(_seg(), ctx)
        assert result.tags["enrolled"] is False

    def test_observe_mode_true_when_no_profile(self, ctx) -> None:
        g = _guard(profile=None)
        result = g.check(_seg(), ctx)
        assert result.tags["observe_mode"] is True


# ---------------------------------------------------------------------------
# Tests: observe mode (threshold = None)
# ---------------------------------------------------------------------------

class TestObserveMode:
    def test_observe_mode_always_passes(self, ctx) -> None:
        cfg = SpeakerConfig(threshold=None, observe_mode=True, min_segment_ms=100)
        g = SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=_anti_profile())
        # Even anti-profile → should pass in observe mode
        result = g.check(_seg(), ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_observe_tag_set(self, ctx) -> None:
        cfg = SpeakerConfig(threshold=None, observe_mode=True, min_segment_ms=100)
        g = SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=_stub_profile())
        result = g.check(_seg(), ctx)
        assert result.tags["observe_mode"] is True

    def test_score_logged_in_observe_mode(self, ctx) -> None:
        cfg = SpeakerConfig(threshold=None, observe_mode=True, min_segment_ms=100)
        g = SpeakerGuard(cfg=cfg, embedder=StubEmbedder(), profile=_stub_profile())
        result = g.check(_seg(), ctx)
        assert result.tags.get("speaker_score") is not None


# ---------------------------------------------------------------------------
# Tests: gating with owner profile (StubEmbedder returns v[0]=1.0)
# ---------------------------------------------------------------------------

class TestOwnerGate:
    def test_owner_passes_above_threshold(self, ctx) -> None:
        g = _guard(profile=_stub_profile(), threshold=0.5)
        result = g.check(_seg(), ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_stranger_dropped_below_threshold(self, ctx) -> None:
        # Anti-profile → cosine score = 0.0 < 0.5
        g = _guard(profile=_anti_profile(), threshold=0.5, observe_mode=False)
        result = g.check(_seg(), ctx)
        assert result.verdict == GuardVerdict.DROP

    def test_score_in_tags(self, ctx) -> None:
        g = _guard(profile=_stub_profile(), threshold=0.5)
        result = g.check(_seg(), ctx)
        assert isinstance(result.tags.get("speaker_score"), float)

    def test_threshold_in_tags(self, ctx) -> None:
        g = _guard(profile=_stub_profile(), threshold=0.5)
        result = g.check(_seg(), ctx)
        assert result.tags["threshold_used"] == pytest.approx(0.5, abs=1e-6)

    def test_drop_reason_contains_score(self, ctx) -> None:
        g = _guard(profile=_anti_profile(), threshold=0.5, observe_mode=False)
        result = g.check(_seg(), ctx)
        assert "score" in result.reason.lower() or "threshold" in result.reason.lower()


# ---------------------------------------------------------------------------
# Tests: short segment skipped
# ---------------------------------------------------------------------------

class TestShortSegment:
    def test_short_segment_passes(self, ctx) -> None:
        g = _guard(profile=_stub_profile(), threshold=0.5)
        short = _seg(speech_ms=50)  # < min_segment_ms=100
        result = g.check(short, ctx)
        assert result.verdict == GuardVerdict.PASS

    def test_short_segment_no_score(self, ctx) -> None:
        g = _guard(profile=_stub_profile(), threshold=0.5)
        result = g.check(_seg(speech_ms=50), ctx)
        assert result.tags.get("speaker_score") is None


# ---------------------------------------------------------------------------
# Tests: guard contract
# ---------------------------------------------------------------------------

class TestGuardContract:
    def test_guard_name(self) -> None:
        assert _guard().name == "g3_speaker"

    def test_fail_closed_on_exception(self, ctx) -> None:
        from unittest.mock import patch
        g = _guard(profile=_stub_profile(), threshold=0.5)
        with patch.object(g._embedder, "embed", side_effect=RuntimeError("embed crash")):
            result = g.check(_seg(), ctx)
        assert result.verdict == GuardVerdict.DROP

    def test_latency_recorded(self, ctx) -> None:
        g = _guard(profile=_stub_profile())
        result = g.check(_seg(), ctx)
        assert result.latency_ms >= 0.0

    def test_reset_does_not_raise(self) -> None:
        g = _guard(profile=_stub_profile())
        g.reset()


# ---------------------------------------------------------------------------
# Tests: CosineSpeakerGate
# ---------------------------------------------------------------------------

class TestCosineSpeakerGate:
    def test_owner_passes(self) -> None:
        g = CosineSpeakerGate(_stub_profile(), threshold=0.5, strict_threshold=0.7)
        r = g.check(_stub_profile())  # score = 1.0
        assert r.accepted is True

    def test_stranger_dropped(self) -> None:
        g = CosineSpeakerGate(_stub_profile(), threshold=0.5, strict_threshold=0.7)
        r = g.check(_anti_profile())  # score = 0.0
        assert r.accepted is False

    def test_strict_threshold_applied_when_assistant_speaking(self) -> None:
        # Score = 0.6 → passes normal (0.5) but fails strict (0.7)
        profile = np.zeros(256, dtype=np.float32)
        profile[0] = 1.0
        emb = np.zeros(256, dtype=np.float32)
        emb[0] = 0.6
        emb[1] = 0.8  # will be L2-normalised inside check
        g = CosineSpeakerGate(profile, threshold=0.5, strict_threshold=0.9)
        r_normal = g.check(emb, assistant_speaking=False)
        r_strict = g.check(emb, assistant_speaking=True)
        assert r_normal.threshold_used == pytest.approx(0.5, abs=1e-6)
        assert r_strict.threshold_used == pytest.approx(0.9, abs=1e-6)

    def test_strict_threshold_never_looser(self) -> None:
        g = CosineSpeakerGate(_stub_profile(), threshold=0.8, strict_threshold=0.3)
        # strict_threshold=0.3 is looser than threshold=0.8 → should be clamped
        assert g.strict_threshold >= g.threshold

    def test_observe_mode_always_accepts(self) -> None:
        g = CosineSpeakerGate(_stub_profile(), threshold=0.9, strict_threshold=0.9,
                              observe_mode=True)
        r = g.check(_anti_profile())  # score 0.0 → would normally fail
        assert r.accepted is True


# ---------------------------------------------------------------------------
# Tests: RollingEmbeddingGate
# ---------------------------------------------------------------------------

class TestRollingEmbeddingGate:
    def test_single_embedding_scored(self) -> None:
        g = RollingEmbeddingGate(_stub_profile(), threshold=0.5, strict_threshold=0.7)
        r = g.push(_stub_profile())
        assert r.accepted is True

    def test_window_accumulation(self) -> None:
        g = RollingEmbeddingGate(_stub_profile(), threshold=0.5, strict_threshold=0.7,
                                 window=3)
        # Push 3 owner embeddings
        for _ in range(3):
            r = g.push(_stub_profile())
        assert r.accepted is True

    def test_reset_clears_history(self) -> None:
        g = RollingEmbeddingGate(_stub_profile(), threshold=0.5, strict_threshold=0.7,
                                 window=3)
        g.push(_stub_profile())
        g.push(_stub_profile())
        g.reset()
        # After reset only the new embedding counts
        r = g.push(_stub_profile())
        assert r.score > 0.0

    def test_strict_threshold_during_barge_in(self) -> None:
        g = RollingEmbeddingGate(_stub_profile(), threshold=0.5, strict_threshold=0.95,
                                 window=1)
        # score ~1.0 → passes both thresholds
        r = g.push(_stub_profile(), assistant_speaking=True)
        assert r.threshold_used == pytest.approx(0.95, abs=1e-6)

    def test_observe_mode(self) -> None:
        g = RollingEmbeddingGate(_anti_profile(), threshold=0.9, strict_threshold=0.9,
                                 observe_mode=True, window=1)
        r = g.push(_stub_profile())  # would score 0.0 → fail normally
        assert r.accepted is True


# ---------------------------------------------------------------------------
# Tests: profile I/O
# ---------------------------------------------------------------------------

class TestProfileIO:
    def test_save_and_load(self, tmp_path) -> None:
        embeddings = [_stub_profile(), _stub_profile()]
        path = str(tmp_path / "owner.npy")
        centroid = save_profile(embeddings, path)
        loaded = load_profile(path)
        assert loaded is not None
        np.testing.assert_allclose(centroid, loaded, atol=1e-5)

    def test_load_missing_returns_none(self, tmp_path) -> None:
        result = load_profile(str(tmp_path / "missing.npy"))
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path) -> None:
        path = str(tmp_path / "nested" / "dir" / "owner.npy")
        save_profile([_stub_profile()], path)
        assert load_profile(path) is not None

    def test_save_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No embeddings"):
            save_profile([], "/tmp/dummy.npy")

    def test_centroid_is_l2_normalised(self, tmp_path) -> None:
        path = str(tmp_path / "owner.npy")
        centroid = save_profile([_stub_profile()], path)
        norm = float(np.linalg.norm(centroid))
        assert abs(norm - 1.0) < 1e-5

    def test_load_returns_l2_normalised(self, tmp_path) -> None:
        path = str(tmp_path / "owner.npy")
        # Save a non-unit vector
        v = np.ones(256, dtype=np.float32) * 5.0
        np.save(path, v)
        loaded = load_profile(path)
        assert loaded is not None
        norm = float(np.linalg.norm(loaded))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: Fbank feature extraction
# ---------------------------------------------------------------------------

class TestFbankExtraction:
    def test_output_shape(self) -> None:
        from voice_firewall.speaker.embedder import extract_fbank
        audio = np.zeros(16000, dtype=np.float32)
        feats = extract_fbank(audio)
        # 1 second @ 10ms hop ≈ 100 frames; we just check dimensions
        assert feats.ndim == 2
        assert feats.shape[1] == 80

    def test_output_dtype(self) -> None:
        from voice_firewall.speaker.embedder import extract_fbank
        feats = extract_fbank(np.zeros(8000, dtype=np.float32))
        assert feats.dtype == np.float32

    def test_silent_audio_finite(self) -> None:
        from voice_firewall.speaker.embedder import extract_fbank
        feats = extract_fbank(np.zeros(16000, dtype=np.float32))
        assert np.all(np.isfinite(feats))

    def test_random_audio_finite(self) -> None:
        from voice_firewall.speaker.embedder import extract_fbank
        rng = np.random.default_rng(0)
        feats = extract_fbank(rng.standard_normal(16000).astype(np.float32))
        assert np.all(np.isfinite(feats))

    def test_cmvn_zero_mean(self) -> None:
        from voice_firewall.speaker.embedder import extract_fbank
        rng = np.random.default_rng(1)
        feats = extract_fbank(rng.standard_normal(16000).astype(np.float32))
        # After mean subtraction, mean per freq bin ≈ 0
        means = feats.mean(axis=0)
        np.testing.assert_allclose(means, np.zeros(80), atol=1e-4)
