import numpy as np
import pytest

from conftest import si_sdr
from speakerlock.config import Config
from speakerlock.extractor import PassthroughExtractor
from speakerlock.interfaces import TargetCondition
from speakerlock.pipeline import SpeakerLock

SR = 16000


class ToneVAD:
    """Speech = frame RMS above 0.01 (deterministic stand-in for Silero)."""
    frame_samples = 512

    def probability(self, frame):
        return 1.0 if np.sqrt(np.mean(frame ** 2)) > 0.01 else 0.0

    def reset(self):
        pass


class SignEmbedder:
    """Positive-mean audio -> 'me', negative-mean -> 'other' (lets tests script who is talking)."""
    dim = 2

    def embed(self, audio):
        return np.array([1.0, 0.0]) if audio.mean() > 0 else np.array([0.0, 1.0])


PROFILES = {"me": np.array([1.0, 0.0]), "other": np.array([0.0, 1.0])}


def make(background=False, **gate):
    cfg = Config.from_dict({"gate": {"fade_ms": 0, "hold_ms": 0, "max_failures": 2, **gate},
                            "verifier": {"window_s": 0.64, "min_speech_s": 0.32, "hop_ms": 160}})
    calls = []
    sl = SpeakerLock(cfg, "me", TargetCondition({}), PROFILES, PassthroughExtractor(), ToneVAD(), SignEmbedder(),
                     on_unverified=calls.append, background_verify=background)
    return sl, calls


def voice(sign, seconds):
    return (sign * (0.1 + 0.05 * np.sin(np.arange(int(seconds * SR)) / 7))).astype(np.float32)


SCRIPT = np.concatenate([voice(+1, 2.0), np.zeros(SR // 2, np.float32), voice(-1, 2.0), voice(+1, 1.5)])


@pytest.mark.parametrize("chunk", [1, 160, 320, 511, 512, 640, 1000, 4800])
def test_output_length_and_content_independent_of_chunk_size(chunk):
    sl, _ = make()
    ref, _ = sl.process_file(SCRIPT, 512)
    sl.reset()
    got, metas = sl.process_file(SCRIPT, chunk)
    assert len(got) == len(ref) == len(SCRIPT)
    np.testing.assert_array_equal(got, ref)
    assert sum(m.samples for m in metas) == len(SCRIPT)
    assert [m.start_sample for m in metas if m.samples] == sorted(m.start_sample for m in metas if m.samples)


def test_passes_target_silences_other_and_fires_callback():
    sl, calls = make()
    y, _ = sl.process_file(SCRIPT, 512)
    t_end, o_start, o_end = 2 * SR, int(2.5 * SR), int(4.5 * SR)
    assert np.allclose(y[SR:t_end], SCRIPT[SR:t_end])            # verified target passes untouched
    assert np.abs(y[o_start + SR // 2:o_end]).max() == 0          # other speaker -> silence
    assert len(calls) == 1 and calls[0]["user_id"] == "me" and calls[0]["best_user"] == "other"
    assert np.abs(y[o_end + SR:]).max() > 0                       # target comes back -> re-verified


def test_first_window_is_withheld_until_verified():
    sl, _ = make()
    y, metas = sl.process_file(voice(+1, 1.0), 512)
    first_open = next(m.start_sample for m in metas if m.verified)
    assert 0.25 * SR <= first_open <= 0.5 * SR                   # ~min_speech_s (0.32 s), then opens
    assert np.abs(y[: int(0.25 * SR)]).max() == 0


def test_start_open_passes_immediately():
    sl, _ = make(start_open=True, strict_turns=False)
    y, _ = sl.process_file(voice(+1, 0.5), 512)
    assert np.abs(y[:512]).max() > 0


def test_meta_fields_and_latency_budget():
    sl, _ = make()
    _, metas = sl.process_file(SCRIPT, 512)
    m = metas[100]
    assert m.samples == 512 and m.latency_ms >= 32 and m.compute_ms >= 0 and isinstance(m.verified, bool)
    assert max(x.latency_ms for x in metas) < 150


def test_background_verification_mode_runs():
    sl, calls = make(background=True)
    import time
    for i in range(0, len(SCRIPT), 512):
        sl.process(SCRIPT[i:i + 512])
        time.sleep(0.002)
    sl.flush()
    sl.close()
    assert calls


# ------------------------------------------------------------------ real models (acceptance-style)
@pytest.fixture(scope="module")
def real_lock(tmp_path_factory, clips, voxlock_extractor):
    from speakerlock.embedder import OnnxDVectorEmbedder
    from speakerlock.enroll import UserStore, enroll
    from speakerlock.vad import SileroVAD
    store = UserStore(tmp_path_factory.mktemp("users"))
    emb, vad = OnnxDVectorEmbedder(), SileroVAD()
    store.save(enroll("me", clips["target_enroll"], voxlock_extractor, emb, vad))
    store.save(enroll("bob", clips["third"], voxlock_extractor, emb, vad))
    me = store.load("me")
    calls = []
    sl = SpeakerLock(Config(), "me", me.condition, store.profiles(), voxlock_extractor, vad, emb,
                     on_unverified=calls.append, background_verify=False)
    return sl, calls


def test_mixed_stream_keeps_only_enrolled_voice(real_lock, clips):
    sl, calls = real_lock
    sl.reset()
    tgt, itf = clips["target_query"], clips["interferer"]
    gap = np.zeros(SR // 2, np.float32)
    noise = np.random.default_rng(0).normal(0, 0.003, 1).astype(np.float32)
    # [target alone] [gap] [interferer alone] [target + interferer overlapped]
    n = len(tgt)
    mix = np.concatenate([tgt, gap, itf, tgt + 0.8 * itf[:n]])
    mix = mix + np.random.default_rng(0).normal(0, 0.003, len(mix)).astype(np.float32)
    y, metas = sl.process_file(mix, 512)
    assert len(y) == len(mix)
    a, b = n + len(gap), n + len(gap) + len(itf)
    # interferer alone (after the first second, once the gate reacts): essentially silent
    itf_level = 10 * np.log10(np.mean(y[a + SR:b] ** 2) / np.mean(mix[a + SR:b] ** 2) + 1e-12)
    assert itf_level < -25, f"interferer only {itf_level:.1f} dB down"
    # overlap: target recovered (far better than the raw mixture) — skip the first second,
    # during which the still-open window contains mostly interferer speech and the gate is shut
    ref = np.concatenate([tgt])
    assert si_sdr(y[b + SR:], ref[SR:]) > si_sdr(mix[b + SR:], ref[SR:]) + 6
    # target alone after verification: kept
    keep = 10 * np.log10(np.mean(y[SR:n] ** 2) / np.mean(mix[SR:n] ** 2))
    assert keep > -6, f"target attenuated {keep:.1f} dB"


def test_live_runner_keeps_up_without_drift(real_lock, clips):
    """Mic-paced input (32 ms chunks in real time) through LiveRunner + background verification.
    Latency must stay bounded (no backlog growth) even when the CPU is throttled; the absolute
    figure is reported by the benchmark, since it depends on the machine's power state."""
    import time
    from speakerlock.stream import LiveRunner
    sl0, _ = real_lock
    sl = SpeakerLock(Config(), "me", sl0.condition, sl0.profiles, sl0.extractor, sl0.vad, sl0.embedder,
                     background_verify=True)
    x = np.concatenate([clips["target_query"], clips["interferer"], clips["target_query"]])
    got = []
    runner = LiveRunner(sl, lambda y, m: got.append(y)).start()
    start = time.perf_counter()
    for i in range(0, len(x), 512):
        time.sleep(max(0.0, start + (i + 512) / SR - time.perf_counter()))
        runner.push(x[i:i + 512])
    runner.stop()
    sl.close()
    lat = np.array(runner.e2e_ms)
    assert sum(len(g) for g in got) == len(x) and runner.dropped == 0
    third = len(lat) // 3
    assert np.median(lat[-third:]) < np.median(lat[:third]) + 60, "latency drifts upward (falling behind)"
    assert np.percentile(lat, 95) < 300
