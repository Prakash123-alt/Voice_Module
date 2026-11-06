"""Low-level numpy DSP helpers.

Adapted from VoiceModule voice/dsp.py (same author).
"""
from __future__ import annotations

import io
import wave

import numpy as np


def to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    return x.astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler.  Good enough for reference/AEC; use a
    higher-quality resampler for speaker embeddings if required by the model."""
    if src_rate == dst_rate or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(x.shape[0] * dst_rate / src_rate))
    src_t = np.arange(x.shape[0], dtype=np.float64)
    dst_t = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    return np.interp(dst_t, src_t, x.astype(np.float64)).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x), dtype=np.float64))) if x.size else 0.0
    return 20.0 * np.log10(r + 1e-9)


def to_wav_bytes(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_int16(x).tobytes())
    return buf.getvalue()


def from_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        rate, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return to_float32(pcm), rate


def estimate_delay_samples(
    reference: np.ndarray, mic: np.ndarray, max_lag: int
) -> tuple[int, float]:
    """Return (lag_samples, confidence) via FFT cross-correlation.

    confidence is the peak-to-mean ratio; < 4.0 is inconclusive.
    """
    n = min(reference.shape[0], mic.shape[0])
    if n < max_lag * 2:
        return 0, 0.0
    r = reference[:n] - reference[:n].mean()
    m = mic[:n] - mic[:n].mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.abs(
        np.fft.irfft(
            np.fft.rfft(m, size) * np.conj(np.fft.rfft(r, size)), size
        )[:max_lag + 1]
    )
    lag = int(np.argmax(corr))
    conf = float(corr[lag] / (corr.mean() + 1e-12))
    return lag, conf


class FrameBuffer:
    """Accumulates arbitrary-length audio and emits fixed-size frames."""

    def __init__(self, frame_samples: int) -> None:
        self.frame_samples = frame_samples
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        n = self._buf.shape[0] // self.frame_samples
        frames = [
            self._buf[i * self.frame_samples : (i + 1) * self.frame_samples]
            for i in range(n)
        ]
        self._buf = self._buf[n * self.frame_samples :]
        return frames

    def flush(self) -> np.ndarray | None:
        if self._buf.size == 0:
            return None
        out = np.zeros(self.frame_samples, dtype=np.float32)
        out[: self._buf.shape[0]] = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        return out
