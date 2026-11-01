"""Audio file / PCM helpers (16 kHz mono float32 everywhere inside the SDK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


def resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return np.asarray(x, np.float32)
    try:
        from math import gcd  # noqa: PLC0415

        from scipy.signal import resample_poly  # noqa: PLC0415
        g = gcd(sr, target)
        return resample_poly(x, target // g, sr // g).astype(np.float32)
    except ImportError:  # linear fallback (fine for speech-band verification, not for hi-fi)
        t = np.arange(int(len(x) * target / sr)) * (sr / target)
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr)


def save_audio(path: str | Path, x: np.ndarray) -> None:
    sf.write(str(path), np.clip(x, -1, 1), SAMPLE_RATE, subtype="PCM_16")


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def from_pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0
