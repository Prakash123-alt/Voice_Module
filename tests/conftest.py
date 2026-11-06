"""Shared pytest fixtures.

No audio hardware, models, or network calls are needed to run this suite.
All heavy dependencies (sounddevice, torch, onnxruntime) are either mocked
or not imported during unit tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from voice_firewall._config import FirewallConfig
from voice_firewall._types import PipelineContext, RawSegment


@pytest.fixture
def cfg() -> FirewallConfig:
    """Default config — no file I/O, all thresholds at their defaults."""
    return FirewallConfig.defaults()


@pytest.fixture
def ctx() -> PipelineContext:
    return PipelineContext()


@pytest.fixture
def silent_segment() -> RawSegment:
    """320-sample silence (20 ms @ 16 kHz)."""
    return RawSegment(audio=np.zeros(320, dtype=np.float32))


@pytest.fixture
def speech_segment() -> RawSegment:
    """320-sample white-noise burst standing in for speech."""
    rng = np.random.default_rng(42)
    return RawSegment(
        audio=(rng.standard_normal(320) * 0.1).astype(np.float32),
        speech_ms=400,
    )


@pytest.fixture
def one_second_audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(16000) * 0.1).astype(np.float32)
