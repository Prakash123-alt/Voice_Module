"""Tests for DSPLoop — real-time DSP pipeline (no hardware required)."""
from __future__ import annotations

import numpy as np
import pytest

from voice_firewall._config import FirewallConfig
from voice_firewall._dsp_loop import DSPLoop
from voice_firewall._types import RawSegment
from voice_firewall.audio.aec import NoAEC
from voice_firewall.audio.loopback import StubLoopback
from voice_firewall.vad.segmenter import EnergyVAD, Segmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(cfg=None, segments=None):
    """DSPLoop with fake/stub backends — no hardware."""
    if cfg is None:
        cfg = FirewallConfig.defaults()
    collected = segments if segments is not None else []

    loop = DSPLoop(
        cfg,
        on_segment=collected.append,
        loopback=StubLoopback(),
        aec=NoAEC(),
        vad=EnergyVAD(threshold_db=-50.0),  # very sensitive — catches any non-silence
    )
    return loop, collected


def _silence(n=320):
    return np.zeros(n, dtype=np.float32)


def _speech(n=320, amp=0.3):
    rng = np.random.default_rng(42)
    return (rng.standard_normal(n) * amp).astype(np.float32)


def _drive(loop: DSPLoop, n_frames: int, frame_fn) -> None:
    """Feed n_frames of audio produced by frame_fn into the loop."""
    for _ in range(n_frames):
        loop.feed_frame(frame_fn())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDSPLoopConstruction:
    def test_builds_with_stub_backends(self) -> None:
        loop, _ = _make_loop()
        assert loop is not None

    def test_start_raises_without_mic(self) -> None:
        loop, _ = _make_loop()
        with pytest.raises(RuntimeError, match="No mic backend"):
            loop.start()

    def test_reset_does_not_raise(self) -> None:
        loop, _ = _make_loop()
        loop.reset()


# ---------------------------------------------------------------------------
# feed_frame() — no hardware needed
# ---------------------------------------------------------------------------

class TestFeedFrame:
    def test_silence_produces_no_segment(self) -> None:
        loop, segments = _make_loop()
        _drive(loop, 50, _silence)
        assert len(segments) == 0

    def test_speech_followed_by_silence_produces_segment(self) -> None:
        loop, segments = _make_loop()
        # Speak for 1 second (50 frames @ 20 ms)
        _drive(loop, 50, _speech)
        # Silence for >500 ms (hangover) → segment emitted
        _drive(loop, 30, _silence)
        assert len(segments) >= 1

    def test_emitted_segment_is_raw_segment(self) -> None:
        loop, segments = _make_loop()
        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        assert isinstance(segments[0], RawSegment)

    def test_emitted_audio_is_float32(self) -> None:
        loop, segments = _make_loop()
        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        assert segments[0].audio.dtype == np.float32

    def test_speech_ms_positive(self) -> None:
        loop, segments = _make_loop()
        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        assert segments[0].speech_ms > 0

    def test_loopback_unavailable_gives_none_suppression(self) -> None:
        """StubLoopback.available is False → suppression_db must be None."""
        loop, segments = _make_loop()
        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        assert segments[0].aec_suppression_db is None

    def test_suppression_db_set_when_loopback_available(self) -> None:
        """When we fake a loopback-available backend, suppression_db is a float."""
        from voice_firewall.audio.loopback import LoopbackCapture

        class FakeLoopback(LoopbackCapture):
            @property
            def available(self):
                return True  # pretend loopback is active
            def start(self): pass
            def stop(self): pass
            def read(self, timeout=None):
                return np.zeros(320, dtype=np.float32)

        cfg = FirewallConfig.defaults()
        segments = []
        loop = DSPLoop(cfg, on_segment=segments.append,
                       loopback=FakeLoopback(), aec=NoAEC(),
                       vad=EnergyVAD(threshold_db=-50.0))

        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        assert segments[0].aec_suppression_db is not None
        assert isinstance(segments[0].aec_suppression_db, float)

    def test_reset_clears_accumulated_energy(self) -> None:
        loop, segments = _make_loop()
        _drive(loop, 20, _speech)
        loop.reset()
        # Energy accumulators should be zeroed
        assert loop._pre_sum == 0.0
        assert loop._post_sum == 0.0
        assert loop._n_frames == 0


# ---------------------------------------------------------------------------
# AEC suppression metric
# ---------------------------------------------------------------------------

class TestAECSuppressionMetric:
    def test_noop_aec_gives_zero_suppression(self) -> None:
        """NoAEC passes audio unchanged → suppression ≈ 0 dB."""
        from voice_firewall.audio.loopback import LoopbackCapture

        class FakeLoopback(LoopbackCapture):
            @property
            def available(self): return True
            def start(self): pass
            def stop(self): pass
            def read(self, timeout=None): return _silence()

        cfg = FirewallConfig.defaults()
        segments = []
        loop = DSPLoop(cfg, on_segment=segments.append,
                       loopback=FakeLoopback(), aec=NoAEC(),
                       vad=EnergyVAD(threshold_db=-50.0))

        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)

        sup = segments[0].aec_suppression_db
        assert sup is not None
        # NoAEC passes audio unchanged → suppression should be ~0 dB
        assert abs(sup) < 1.0

    def test_multiple_segments_independent_suppression(self) -> None:
        """Each segment has its own independent suppression measurement."""
        loop, segments = _make_loop()
        # First utterance
        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        # Second utterance
        _drive(loop, 50, _speech)
        _drive(loop, 30, _silence)
        assert len(segments) >= 2
        # Each segment's measurement is independent (None for StubLoopback)
        for seg in segments:
            assert seg.aec_suppression_db is None  # StubLoopback
