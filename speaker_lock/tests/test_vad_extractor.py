import time

import numpy as np
import pytest

from conftest import requires_voxlock, si_sdr
from speakerlock.extractor import PassthroughExtractor, loudest
from speakerlock.vad import EnergyVAD, SileroVAD


def _frame_probs(vad, x, n=512):
    return np.array([vad.probability(x[i:i + n]) for i in range(0, len(x) - n + 1, n)])


def test_silero_separates_speech_from_noise(clips):
    vad = SileroVAD(512)
    speech = _frame_probs(vad, clips["target_query"])
    vad.reset()
    noise = _frame_probs(vad, np.random.default_rng(0).normal(0, 0.01, 48000).astype(np.float32))
    assert speech.max() > 0.9 and np.mean(speech > 0.5) > 0.4
    assert noise.max() < 0.3


def test_silero_reblocks_non_native_frames(clips):
    a = _frame_probs(SileroVAD(512), clips["target_query"], 512)
    b = _frame_probs(SileroVAD(640), clips["target_query"], 640)   # 40 ms frames
    assert np.mean(a > 0.5) == pytest.approx(np.mean(b > 0.5), abs=0.1)


def test_energy_vad_basic():
    vad = EnergyVAD(512)
    quiet = np.random.default_rng(0).normal(0, 0.001, 512 * 40).astype(np.float32)
    _frame_probs(vad, quiet)
    loud = np.sin(np.arange(512) / 5).astype(np.float32) * 0.3
    assert vad.probability(loud) > 0.9


def test_loudest_picks_3s():
    x = np.zeros(16000 * 6, np.float32)
    x[16000 * 2:16000 * 5] = 0.5
    y = loudest(x, 3.0)
    assert len(y) == 48000 and y.min() == 0.5


def test_passthrough_identity():
    ex = PassthroughExtractor()
    ex.start(ex.condition(np.zeros(16000)))
    x = np.random.default_rng(1).normal(size=1000).astype(np.float32)
    np.testing.assert_array_equal(ex.process(x), x)


def _stream(ex, cond, x, chunk):
    ex.start(cond)
    out = [ex.process(x[i:i + chunk]) for i in range(0, len(x), chunk)]
    return np.concatenate(out + [ex.flush()])


@pytest.fixture(scope="module")
def fp32_extractor():
    from speakerlock.extractor import VoxLockExtractor
    return VoxLockExtractor("fp32", threads=2)


def _mix(clips):
    return (clips["target_query"][:16000 * 2] + 0.5 * clips["interferer"][:16000 * 2]).astype(np.float32)


@requires_voxlock
@pytest.mark.parametrize("chunk", [1, 100, 512, 777])
def test_voxlock_chunk_boundaries_exact_fp32(fp32_extractor, clips, chunk):
    """fp32: identical output for any chunking (only LSTM state crosses calls); len(out) == len(in)."""
    ex = fp32_extractor
    cond, x = ex.condition(clips["target_enroll"]), _mix(clips)
    ref, got = _stream(ex, cond, x, 4096), _stream(ex, cond, x, chunk)
    assert len(ref) == len(got) == len(x)
    np.testing.assert_allclose(got, ref, atol=1e-5)


@requires_voxlock
@pytest.mark.parametrize("chunk", [1, 512, 777])
def test_voxlock_chunk_boundaries_int8(voxlock_extractor, clips, chunk):
    """int8 uses dynamic quantisation whose scale depends on frames per call: allow inaudible differences."""
    ex = voxlock_extractor
    cond, x = ex.condition(clips["target_enroll"]), _mix(clips)
    ref, got = _stream(ex, cond, x, 4096), _stream(ex, cond, x, chunk)
    assert len(ref) == len(got) == len(x)
    snr = 10 * np.log10(np.sum(ref ** 2) / (np.sum((got - ref) ** 2) + 1e-12))
    assert snr > 25, f"chunking changed the output too much ({snr:.1f} dB)"


@requires_voxlock
def test_voxlock_isolates_target(voxlock_extractor, clips):
    ex = voxlock_extractor
    tgt, itf = clips["target_query"], clips["interferer"][: len(clips["target_query"])]
    mix = (tgt + itf).astype(np.float32)
    y = _stream(ex, ex.condition(clips["target_enroll"]), mix, 512)
    gain = si_sdr(y, tgt) - si_sdr(mix, tgt)
    assert gain > 6.0, f"SI-SDR improvement only {gain:.1f} dB"


@requires_voxlock
def test_voxlock_suppresses_non_target_alone(voxlock_extractor, clips):
    ex = voxlock_extractor
    itf = clips["interferer"]
    y = _stream(ex, ex.condition(clips["target_enroll"]), itf, 512)
    atten_db = 10 * np.log10(np.mean(itf ** 2) / (np.mean(y ** 2) + 1e-12))
    assert atten_db > 10, f"interferer only attenuated {atten_db:.1f} dB"


@requires_voxlock
def test_voxlock_realtime_32ms(voxlock_extractor, clips):
    """Sanity bound on per-chunk cost. This laptop needs ~38-45 ms per 32 ms frame
    (RTF slightly above 1), so 40-80 ms chunks are recommended here; the assert only
    guards against pathological regressions, not the real-time budget itself."""
    ex = voxlock_extractor
    ex.start(ex.condition(clips["target_enroll"]))
    x, ts = clips["target_query"], []
    for i in range(0, len(x) - 512, 512):
        t0 = time.perf_counter()
        ex.process(x[i:i + 512])
        ts.append((time.perf_counter() - t0) * 1000)
    assert np.median(ts[3:]) < 64, f"median {np.median(ts[3:]):.1f} ms per 32 ms frame"
