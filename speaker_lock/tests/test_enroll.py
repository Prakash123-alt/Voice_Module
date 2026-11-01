import numpy as np
import pytest

from speakerlock.enroll import EnrollmentError, UserStore, check_clip, enroll
from speakerlock.extractor import PassthroughExtractor
from speakerlock.interfaces import TargetCondition, cosine
from speakerlock.vad import SileroVAD


class MeanEmbedder:
    dim = 3

    def embed(self, audio):
        return np.array([audio.mean(), audio.std(), 1.0], np.float32)


def test_check_clip_rejects_short_and_silent():
    with pytest.raises(EnrollmentError, match="at least"):
        check_clip(np.ones(16000 * 2, np.float32) * 0.1)
    with pytest.raises(EnrollmentError, match="too quiet"):
        check_clip(np.zeros(16000 * 5, np.float32))
    assert any("5 s" in w for w in check_clip(np.full(16000 * 3, 0.1, np.float32)))
    assert check_clip(np.full(16000 * 5, 0.1, np.float32)) == []


def test_enroll_rejects_clip_without_speech():
    noise = np.random.default_rng(0).normal(0, 0.05, 16000 * 5).astype(np.float32)
    with pytest.raises(EnrollmentError, match="speech detected"):
        enroll("me", noise, PassthroughExtractor(), MeanEmbedder(), SileroVAD())


def test_enroll_rejects_bad_user_id(clips):
    with pytest.raises(EnrollmentError, match="user_id"):
        enroll("../evil", clips["target_enroll"], PassthroughExtractor(), MeanEmbedder())


def test_store_roundtrip_multi_user(tmp_path, clips):
    store = UserStore(tmp_path)
    a = enroll("alice", clips["target_enroll"], PassthroughExtractor(), MeanEmbedder(), SileroVAD())
    a.condition = TargetCondition({"enroll_mag": np.ones((1, 257, 5), np.float32)})
    b = enroll("bob", clips["interferer"], PassthroughExtractor(), MeanEmbedder(), SileroVAD())
    store.save(a), store.save(b)
    assert store.users() == ["alice", "bob"]
    got = store.load("alice")
    np.testing.assert_array_equal(got.dvector, a.dvector)
    np.testing.assert_array_equal(got.condition.arrays["enroll_mag"], a.condition.arrays["enroll_mag"])
    assert got.meta["speech_s"] > 1.5 and set(store.profiles()) == {"alice", "bob"}
    store.delete("bob")
    assert store.users() == ["alice"]
    with pytest.raises(KeyError):
        store.load("bob")


def test_real_enrollment_profile_matches_same_speaker(clips, voxlock_extractor):
    from speakerlock.embedder import OnnxDVectorEmbedder
    emb, vad = OnnxDVectorEmbedder(), SileroVAD()
    me = enroll("me", clips["target_enroll"], voxlock_extractor, emb, vad)
    assert set(me.condition.arrays) == {"enroll_mag", "context"}
    same = cosine(me.dvector, emb.embed(clips["target_query"]))
    other = cosine(me.dvector, emb.embed(clips["interferer"]))
    assert same > 0.7 and same > other + 0.1, (same, other)
