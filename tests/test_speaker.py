import numpy as np
import pytest

from voice.speaker import (Embedder, ProfileSpeakerVerifier, average_embedding, cosine_similarity,
                           suggest_threshold)


class FixedEmbedder(Embedder):
    """Returns a preset embedding regardless of audio, so threshold logic is tested in isolation."""

    def __init__(self, emb):
        self.emb = np.asarray(emb, dtype=np.float64)  # float64 so the 0.65 boundary case is exact
        self.calls = 0

    def embed(self, audio):
        self.calls += 1
        return self.emb


def unit_at(cos: float) -> np.ndarray:
    """2-D unit vector with the given cosine to [1, 0]."""
    return np.array([cos, np.sqrt(1 - cos ** 2)], dtype=np.float64)


AUDIO = np.zeros(16000, dtype=np.float32)  # 1 s


def test_cosine_basic():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0  # no div-by-zero


@pytest.mark.parametrize("score,accepted", [(0.80, True), (0.65, True), (0.6499, False), (0.30, False)])
def test_threshold_boundary(score, accepted):
    v = ProfileSpeakerVerifier(FixedEmbedder(unit_at(score)), np.array([1, 0]), threshold=0.65)
    v.profile = np.array([1.0, 0.0])
    res = v.verify(AUDIO)
    assert res.accepted is accepted
    assert res.score == pytest.approx(score, abs=1e-4)
    assert res.threshold == 0.65


def test_short_audio_rejected_without_embedding():
    emb = FixedEmbedder(unit_at(0.99))
    v = ProfileSpeakerVerifier(emb, np.array([1, 0]), threshold=0.65, min_segment_ms=400)
    res = v.verify(np.zeros(16000 * 300 // 1000, dtype=np.float32))
    assert not res.accepted and emb.calls == 0


def test_min_samples_override_allows_short_barge_in_window():
    v = ProfileSpeakerVerifier(FixedEmbedder(unit_at(0.9)), np.array([1, 0]), threshold=0.65, min_segment_ms=400)
    window = np.zeros(16000 * 260 // 1000, dtype=np.float32)  # 260 ms
    assert not v.verify(window).accepted
    assert v.verify(window, min_samples=16000 * 250 // 1000).accepted


def test_profile_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="enroll.py"):
        ProfileSpeakerVerifier.from_file(FixedEmbedder([1, 0]), tmp_path / "nope.npy")


def test_profile_roundtrip(tmp_path):
    p = tmp_path / "voice.npy"
    np.save(p, np.array([1, 0], dtype=np.float32))
    v = ProfileSpeakerVerifier.from_file(FixedEmbedder(unit_at(0.7)), p, threshold=0.65)
    assert v.verify(AUDIO).accepted


def test_average_embedding_is_unit_and_scale_invariant():
    c = average_embedding([np.array([10.0, 0.0]), np.array([0.0, 0.1])])
    assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-6)
    assert c[0] == pytest.approx(c[1], abs=1e-6)  # each input normalised before averaging


def test_take_check_ignores_single_click_but_catches_real_clipping():
    from enroll import check_take
    rng = np.random.default_rng(0)
    speech = (0.1 * rng.standard_normal(64000)).astype(np.float32)
    click = speech.copy(); click[:80] = 1.0                       # stream-open pop: 80 samples
    assert check_take(click)[0] is True
    clipped = np.clip(speech * 30, -1, 1)                         # genuinely overdriven
    assert check_take(clipped)[1] == "clipping"
    assert check_take(speech * 0.001)[1] == "too_quiet"


def test_suggest_threshold_genuine_only():
    rep = suggest_threshold([0.85, 0.88, 0.90, 0.86, 0.87])
    assert rep["suggested"] < rep["genuine_min"]
    assert 0.3 <= rep["suggested"] <= 0.85


def test_suggest_threshold_floor():
    assert suggest_threshold([0.2, 0.25, 0.3])["suggested"] == 0.3


def test_suggest_threshold_with_impostors_separates():
    genuine = [0.82, 0.85, 0.88, 0.90]
    impostor = [0.30, 0.42, 0.50, 0.55]
    rep = suggest_threshold(genuine, impostor)
    assert 0.55 <= rep["suggested"] <= 0.82
    assert rep["eer"] == 0.0
