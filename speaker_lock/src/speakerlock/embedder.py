"""Speaker embedders (d-vector). Default: the GE2E encoder from Resemblyzer (Apache-2.0,
https://github.com/resemble-ai/Resemblyzer) exported to ONNX, with its librosa mel front-end ported to
numpy so verification needs neither torch nor librosa at runtime."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, VerifierConfig
from .interfaces import Embedder, l2norm
from .ort_utils import session

N_FFT, HOP, N_MELS, PARTIAL = 400, 160, 40, 160      # 25 ms window, 10 ms hop, 40 mels, 1.6 s partials
TARGET_DBFS = -30.0


def _hz_to_mel(f: np.ndarray) -> np.ndarray:            # Slaney scale (librosa htk=False)
    f = np.asarray(f, np.float64)
    m = f / (200.0 / 3)
    log = f >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), m)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * (200.0 / 3))


@lru_cache(maxsize=4)
def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Equivalent to librosa.filters.mel(sr, n_fft, n_mels) (Slaney scale, Slaney norm)."""
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff, ramps = np.diff(mel_f), mel_f[:, None] - fft_f[None, :]
    w = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    w *= (2.0 / (mel_f[2:] - mel_f[:-2]))[:, None]
    return w.astype(np.float32)


_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic


def mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """Power mel spectrogram, (frames, 40) - matches librosa.feature.melspectrogram (center=True, zero pad)."""
    x = np.pad(np.asarray(wav, np.float32), (N_FFT // 2, N_FFT // 2))
    n = 1 + (x.shape[0] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _HANN, axis=1)) ** 2
    return (spec @ mel_filterbank().T).astype(np.float32)


def normalize_volume(wav: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Raise (never lower) the level to target_dbfs, as Resemblyzer's preprocessing does."""
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    if rms <= 1e-8:
        return wav
    change = target_dbfs - 20 * np.log10(rms)
    return wav * (10 ** (change / 20)) if change > 0 else wav


def partial_slices(n_samples: int, rate: float = 1.3, min_coverage: float = 0.75) -> list[slice]:
    """Mel-frame slices of the 1.6 s partial utterances (port of VoiceEncoder.compute_partial_slices)."""
    n_frames = int(np.ceil((n_samples + 1) / HOP))
    step = int(np.round((SAMPLE_RATE / rate) / HOP))
    slices = [slice(i, i + PARTIAL) for i in range(0, max(1, n_frames - PARTIAL + step + 1), step)]
    last = slices[-1]
    if len(slices) > 1 and (n_samples - last.start * HOP) / (PARTIAL * HOP) < min_coverage:
        slices = slices[:-1]
    return slices


def mel_partials(wav: np.ndarray) -> np.ndarray:
    """(n_partials, 160, 40) batch for the encoder, zero-padding the waveform as Resemblyzer does."""
    wav = normalize_volume(np.asarray(wav, np.float32))
    slices = partial_slices(wav.shape[0])
    need = slices[-1].stop * HOP
    if need >= wav.shape[0]:
        wav = np.pad(wav, (0, need - wav.shape[0]))
    mel = mel_spectrogram(wav)
    return np.stack([mel[s] for s in slices])


class OnnxDVectorEmbedder(Embedder):
    """GE2E d-vector (256-d) on onnxruntime."""

    dim = 256

    def __init__(self, model_path: Optional[str] = None, threads: int = 1, device: str = "cpu") -> None:
        from .models import dvector  # noqa: PLC0415
        self._sess = session(dvector(model_path), threads=threads, device=device)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        partials = self._sess.run(None, {"mels": mel_partials(audio)})[0]
        return l2norm(partials.mean(axis=0))


class ResemblyzerEmbedder(Embedder):
    """Reference implementation on torch (same model, used to validate the ONNX export)."""

    dim = 256

    def __init__(self, device: str = "cpu") -> None:
        import torch  # noqa: PLC0415
        from resemblyzer import VoiceEncoder  # noqa: PLC0415
        self._torch = torch
        self._enc = VoiceEncoder(device=device, verbose=False)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            p = self._enc(self._torch.from_numpy(mel_partials(audio))).cpu().numpy()
        return l2norm(p.mean(axis=0))


def build_embedder(cfg: VerifierConfig, device: str = "cpu") -> Embedder:
    if cfg.backend == "resemblyzer":
        return ResemblyzerEmbedder(device)
    return OnnxDVectorEmbedder(cfg.model_path, cfg.threads, device)
