import numpy as np
import pytest

from speakerlock.config import GateConfig, VerifierConfig
from speakerlock.embedder import OnnxDVectorEmbedder, mel_filterbank, mel_spectrogram
from speakerlock.verifier import HysteresisGate, RollingVerifier, Score

F = 512  # 32 ms frames


# ------------------------------------------------------------------ gate logic (no models)
def S(target, best="me"):
    return Score(target=target, best_user=best)


def gate(**kw):
    calls = []
    g = HysteresisGate(GateConfig(fade_ms=0, **kw), "me", on_unverified=calls.append)
    return g, calls


def test_opens_only_above_open_threshold():
    g, _ = gate(open_threshold=0.7, close_threshold=0.6)
    g.update(S(0.65), 0)
    assert not g.open
    g.update(S(0.71), F)
    assert g.open


def test_hysteresis_keeps_open_between_thresholds():
    g, _ = gate(open_threshold=0.7, close_threshold=0.6)
    g.update(S(0.75), 0)
    g.update(S(0.62), F * 10)           # below open, above close -> still a pass
    assert g.open and g.failures == 0


def test_hold_prevents_chopping_then_closes():
    g, _ = gate(open_threshold=0.7, close_threshold=0.6, hold_ms=400, max_failures=99)
    g.update(S(0.8), 0)
    g.update(S(0.3), 16 * 100)          # failing score 100 ms after the last pass
    assert g.open                        # still inside the 400 ms hold
    g.update(None, 16 * 399)
    assert g.open
    g.update(None, 16 * 400)
    assert not g.open


def test_pass_during_hold_cancels_close():
    g, _ = gate(open_threshold=0.7, close_threshold=0.6, hold_ms=400, max_failures=99)
    g.update(S(0.8), 0)
    g.update(S(0.3), 1600)
    g.update(S(0.65), 3200)
    g.update(None, 16 * 2000)
    assert g.open


def test_n_failures_forces_silence_and_fires_callback_once():
    g, calls = gate(open_threshold=0.7, close_threshold=0.6, hold_ms=10_000, max_failures=3)
    g.update(S(0.9), 0)
    for i in range(1, 6):
        g.update(S(0.2, best="other"), i * F)
    assert not g.open and len(calls) == 1 and calls[0]["failures"] == 3
    g.update(S(0.9), 10 * F)             # recovering resets the streak
    assert g.open and g.failures == 0
    for i in range(11, 15):
        g.update(S(0.1), i * F)
    assert len(calls) == 2


def test_closed_set_rejects_when_other_user_matches_better():
    g, _ = gate(open_threshold=0.7, close_threshold=0.6)
    g.update(Score(0.8, "bob", {"me": 0.8, "bob": 0.85}), 0)
    assert not g.open
    g2 = HysteresisGate(GateConfig(open_threshold=0.7, close_threshold=0.6, closed_set=False), "me")
    g2.update(Score(0.8, "bob"), 0)
    assert g2.open


def test_fade_is_click_free_and_reaches_target():
    g = HysteresisGate(GateConfig(fade_ms=10), "me")       # 160-sample ramp
    ones = np.ones(F, np.float32)
    g.update(S(0.95), 0)
    y = g.apply(ones)
    assert y[0] < 0.01 and np.all(np.diff(y) >= 0) and y[159] == pytest.approx(1.0) and y[-1] == 1.0
    g.update(S(0.1, "x"), F)
    for _ in range(3):
        g.update(S(0.1, "x"), F)
    y = g.apply(ones)
    assert np.all(np.diff(y) <= 0) and y[-1] == 0.0
    assert not g.apply(ones).any()


def test_invert_passes_everyone_except_the_target():
    g = HysteresisGate(GateConfig(open_threshold=0.7, fade_ms=0, invert=True), "me")
    ones = np.ones(4, np.float32)
    g.update(None, 0)                          # nothing verified: audio passes
    assert g.apply(ones).all()
    g.update(S(0.9), F)                        # target verified: now silenced
    assert not g.apply(ones).any()
    g.update(S(0.1, "x"), 2 * F)               # someone else: audio passes again
    g.update(S(0.1, "x"), 3 * F)
    g.update(S(0.1, "x"), 4 * F)
    assert g.apply(ones).all()


def test_start_open_policy():
    g = HysteresisGate(GateConfig(start_open=True, fade_ms=0), "me")
    assert g.open and g.apply(np.ones(4, np.float32)).all()


def test_strict_turns_close_then_reopen_on_pass():
    g, _ = gate(open_threshold=0.7, strict_turns=True)
    g.update(S(0.8), 0)                    # verified turn one
    g.new_turn(10 * F)                     # a pause ended the turn: next talker must re-verify
    assert not g.open
    g.update(None, 15 * F, talking=True)
    assert not g.open
    g.update(S(0.9), 20 * F)
    assert g.open


def test_turn_grace_expires_without_confirming_score():
    g, _ = gate(open_threshold=0.7, turn_grace_ms=1000, strict_turns=False)
    g.update(S(0.8), 0)                    # verified turn one
    g.new_turn(10 * F)
    g.update(None, 10 * F + 16 * 999, talking=True)
    assert g.open                          # still inside the 1 s grace
    g.update(None, 10 * F + 16 * 1000, talking=True)
    assert not g.open                      # no confirming score -> silence
    # a passing score reopens mid-turn
    g.update(S(0.9), 10 * F + 16 * 1200)
    assert g.open


def test_score_ttl_expires_mid_turn_but_not_during_grace():
    g, _ = gate(open_threshold=0.7, score_ttl_ms=2000, turn_grace_ms=1000, strict_turns=False)
    g.update(S(0.8), 0)
    g.update(S(0.8), 5 * F)                # last pass at 5*F
    g.update(None, 5 * F + 16 * 1999, talking=True)
    assert g.open
    g.update(None, 5 * F + 16 * 2000, talking=True)
    assert not g.open                      # no passing score for 2 s while someone talks
    # resume-after-pause is a new turn: governed by the grace window, not the stale TTL
    g2, _ = gate(open_threshold=0.7, score_ttl_ms=2000, turn_grace_ms=1000, strict_turns=False)
    g2.update(S(0.8), 0)
    g2.new_turn(100 * F)
    g2.update(None, 100 * F + 16 * 500, talking=True)
    assert g2.open                         # inside grace even though the old pass is > ttl old


def test_grace_gain_attenuates_unverified_turn_start():
    g = HysteresisGate(GateConfig(open_threshold=0.7, fade_ms=0, grace_gain=0.1, strict_turns=False), "me")
    g.update(S(0.8), 0)
    assert g.apply(np.ones(4, np.float32)).mean() == pytest.approx(1.0)
    g.new_turn(10 * F)
    g.update(None, 11 * F, talking=True)
    assert g.open and g.apply(np.ones(4, np.float32)).mean() == pytest.approx(0.1)
    g.update(S(0.9), 15 * F)
    assert g.apply(np.ones(4, np.float32)).mean() == pytest.approx(1.0)


# ------------------------------------------------------------------ rolling window
class FakeEmbedder:
    dim = 2

    def __init__(self):
        self.calls = []

    def embed(self, audio):
        self.calls.append(len(audio))
        return np.array([audio.mean(), 1.0], np.float32)


def _verifier(**kw):
    cfg = VerifierConfig(window_s=1.0, min_speech_s=0.5, hop_ms=320, max_age_s=2.0, **kw)
    emb = FakeEmbedder()
    profiles = {"me": np.array([0.0, 1.0]), "other": np.array([1.0, 0.0])}
    return RollingVerifier(emb, profiles, "me", cfg, background=False), emb


def test_no_score_until_min_speech_and_only_speech_counts():
    v, emb = _verifier()
    t = 0
    for _ in range(20):                       # silence never triggers a score
        t += F
        assert v.push(np.zeros(F, np.float32), False, t) is None
    scores = []
    for _ in range(40):
        t += F
        s = v.push(np.zeros(F, np.float32), True, t)
        if s:
            scores.append(s)
    assert emb.calls[0] >= 6400                # first score after the new-turn floor (0.4 s of speech)
    assert max(emb.calls) <= 16000 + F         # window capped at 1.0 s
    assert len(scores) == len(emb.calls) and scores[0].best_user == "me"
    gaps = np.diff([s.at_sample for s in scores])
    assert np.all(gaps >= 320 * 16)            # at most one score per hop


def test_old_speech_ages_out():
    v, _ = _verifier()
    for i in range(1, 30):
        v.push(np.zeros(F, np.float32), True, i * F)
    assert not v.idle
    v.push(np.zeros(F, np.float32), False, 29 * F + 2 * 16000 + F)
    assert v.idle


def test_background_scoring_does_not_block():
    cfg = VerifierConfig(window_s=1.0, min_speech_s=0.5, hop_ms=320)
    v = RollingVerifier(FakeEmbedder(), {"me": np.array([0.0, 1.0])}, "me", cfg, background=True)
    import time
    got = []
    for i in range(1, 80):
        s = v.push(np.zeros(F, np.float32), True, i * F)
        if s:
            got.append(s)
        time.sleep(0.001)
    v.close()
    assert got


def test_unknown_target_rejected():
    with pytest.raises(KeyError):
        RollingVerifier(FakeEmbedder(), {"a": np.ones(2)}, "b", VerifierConfig())


# ------------------------------------------------------------------ embedder
def test_mel_front_end_matches_librosa():
    librosa = pytest.importorskip("librosa")
    np.testing.assert_allclose(mel_filterbank(), librosa.filters.mel(sr=16000, n_fft=400, n_mels=40), atol=1e-6)
    x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    ref = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=400, hop_length=160, n_mels=40).T
    np.testing.assert_allclose(mel_spectrogram(x), ref, rtol=1e-3, atol=1e-5)


def test_onnx_dvector_matches_resemblyzer_and_separates_speakers(clips):
    pytest.importorskip("resemblyzer")
    from speakerlock.embedder import ResemblyzerEmbedder
    onnx, ref = OnnxDVectorEmbedder(), ResemblyzerEmbedder()
    for name in ("target_query", "interferer"):
        assert float(np.dot(onnx.embed(clips[name]), ref.embed(clips[name]))) > 0.9999
    enrolled = onnx.embed(clips["target_enroll"])
    same = float(np.dot(enrolled, onnx.embed(clips["target_query"])))
    other = max(float(np.dot(enrolled, onnx.embed(clips[n]))) for n in ("interferer", "third"))
    assert same > other + 0.1, (same, other)
