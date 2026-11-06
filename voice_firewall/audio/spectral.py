"""Spectral analysis utilities for Phase 3 anomaly detection.

near_ultrasound_ratio()
    Returns the fraction of energy in the top band of the spectrum.
    At 16 kHz (Nyquist 8 kHz) the "near-ultrasound proxy" band is 6–8 kHz.
    At 44100 Hz the band covers true near-ultrasound (16–22 kHz).

background_energy_ratio()
    Returns the fraction of energy *outside* the speech band (300–3400 Hz).
    A high value while speech is present suggests hidden audio injection.
"""
from __future__ import annotations

import numpy as np


# Speech band boundaries (Hz)
_SPEECH_LOW_HZ = 300.0
_SPEECH_HIGH_HZ = 3400.0

# Near-ultrasound proxy: top fraction of the spectrum to inspect
_ULTRASOUND_BAND_FRACTION = 0.20   # top 20% of Nyquist range


def _fft_band_energy(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> float:
    """Return the fraction of total FFT energy in [low_hz, high_hz]."""
    n = len(audio)
    if n < 32:
        return 0.0
    fft = np.fft.rfft(audio)
    power = np.abs(fft) ** 2
    total = power.sum()
    if total < 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(power[mask].sum() / total)


def near_ultrasound_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Return the fraction of energy in the top-band proxy for near-ultrasound.

    Parameters
    ----------
    audio:       Float32 mono array.
    sample_rate: Samples per second.  16000 → band is 6400–8000 Hz (proxy).
                 44100 → band is 17640–22050 Hz (true near-ultrasound).

    Returns
    -------
    Float in [0, 1]. Higher means more energy in the high-frequency band.
    """
    nyquist = sample_rate / 2.0
    low_hz = nyquist * (1.0 - _ULTRASOUND_BAND_FRACTION)
    return _fft_band_energy(audio, sample_rate, low_hz, nyquist)


def background_energy_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Return the fraction of energy *outside* the speech band (300–3400 Hz).

    Parameters
    ----------
    audio:       Float32 mono array.
    sample_rate: Samples per second.

    Returns
    -------
    Float in [0, 1]. Higher values suggest hidden audio alongside speech.
    Typical clean speech: < 0.30. Suspicious: > 0.60.
    Returns 0.0 for silent/near-silent audio (nothing to flag).
    """
    if audio.size < 32:
        return 0.0
    total = float((np.abs(np.fft.rfft(audio)) ** 2).sum())
    if total < 1e-12:
        return 0.0  # silence — no energy to analyse
    speech_energy = _fft_band_energy(audio, sample_rate, _SPEECH_LOW_HZ, _SPEECH_HIGH_HZ)
    return 1.0 - speech_energy
