"""ONNX speaker embedder — WeSpeaker ResNet34 (Phase 1 Step 4).

Model: ``onnx-community/wespeaker-voxceleb-resnet34-LM`` (Apache-2.0, 256-dim, fbank80)
  File: ``onnx/model.onnx``

Front-end: 80-dim log-mel filterbank (HTK scale) with pre-emphasis, Hamming window,
  CMVN (utterance-level mean subtraction) — matching WeSpeaker's default kaldi-style
  feature extraction.  Pure NumPy — no librosa, no torch at inference time.

Design
------
- Resemblyzer is explicitly NOT used (declared licence: UNKNOWN).
- Model is downloaded via ``huggingface_hub`` and cached in the HuggingFace hub cache.
- The embedder is called from a **worker thread**, never the DSP thread.
- Output is an L2-normalised float32 numpy vector of dimension 256.
- ``StubEmbedder`` returns a fixed vector for unit tests (no network/model needed).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

import numpy as np

from voice_firewall._logging import log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class SpeakerEmbedder(ABC):
    """Interface for all speaker embedding backends."""

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return an L2-normalised float32 embedding vector of shape (dim,)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""


# ---------------------------------------------------------------------------
# Fbank feature extraction (NumPy only)
# ---------------------------------------------------------------------------

def _hz_to_mel_htk(f: np.ndarray) -> np.ndarray:
    """HTK mel scale (used by Kaldi/WeSpeaker: mel = 2595 * log10(1 + f/700))."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def _mel_filterbank_matrix(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
) -> np.ndarray:
    """Triangular mel filterbank, (n_mels, 1 + n_fft//2), HTK scale."""
    low_mel = _hz_to_mel_htk(np.array(low_freq))
    high_mel = _hz_to_mel_htk(np.array(high_freq))
    mel_points = np.linspace(float(low_mel), float(high_mel), n_mels + 2)
    hz_points = _mel_to_hz_htk(mel_points)
    # Map Hz values to FFT bin indices
    bin_pts = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if mid > lo:
            fbank[m, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fbank[m, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fbank


def extract_fbank(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    win_ms: int = 25,
    hop_ms: int = 10,
    low_freq: float = 20.0,
    high_freq: float = 7600.0,
    pre_emphasis: float = 0.97,
) -> np.ndarray:
    """Compute log-mel filterbank features (Fbank) matching WeSpeaker/Kaldi preprocessing.

    Returns
    -------
    np.ndarray of shape ``(T, n_mels)``, float32, with utterance-level CMVN applied
    (mean subtracted per frequency bin).
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    win_len = int(sr * win_ms / 1000)   # 400 at 16 kHz
    hop_len = int(sr * hop_ms / 1000)   # 160 at 16 kHz

    # Pre-emphasis
    if pre_emphasis > 0.0:
        audio = np.concatenate([[audio[0]], audio[1:] - pre_emphasis * audio[:-1]])

    # Hamming window
    window = np.hamming(win_len).astype(np.float32)

    # Reflect-pad so the first and last frames are centred on the speech
    pad = win_len // 2
    audio = np.pad(audio, (pad, pad + n_fft))

    # Framing → power spectrum
    n_frames = max(1, 1 + (len(audio) - win_len) // hop_len)
    idx = np.arange(win_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    frames = audio[idx] * window                              # (T, win_len)
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # (T, n_fft//2+1)

    # Mel filterbank → log compression
    fb = _mel_filterbank_matrix(sr, n_fft, n_mels, low_freq, high_freq)
    mel_spec = spec @ fb.T                                    # (T, n_mels)
    log_mel = np.log(np.maximum(mel_spec, 1e-10)).astype(np.float32)

    # CMVN: subtract utterance-level mean per frequency bin
    log_mel -= log_mel.mean(axis=0, keepdims=True)
    return log_mel


def _l2norm(v: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(v.astype(np.float64))
    return (v.astype(np.float32) / (float(nrm) + 1e-12))


# ---------------------------------------------------------------------------
# Stub (tests / no model)
# ---------------------------------------------------------------------------

class StubEmbedder(SpeakerEmbedder):
    """Returns a reproducible unit vector.  Used in tests and when no model is available."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        v[0] = 1.0
        return v

    @property
    def dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# ONNX WeSpeaker embedder
# ---------------------------------------------------------------------------

class OnnxWeSpeakerEmbedder(SpeakerEmbedder):
    """WeSpeaker ResNet34-LM ONNX embedder (256-dim).

    Downloads model from ``onnx-community/wespeaker-voxceleb-resnet34-LM``
    on first use; cached in the HuggingFace hub cache directory.

    Parameters
    ----------
    model_path : optional explicit path to a local ``.onnx`` file.
    threads : OnnxRuntime inter/intra-op thread count.
    """

    HF_REPO = "onnx-community/wespeaker-voxceleb-resnet34-LM"
    HF_FILE = "onnx/model.onnx"
    _EMBED_DIM = 256
    _FBANK_DIM = 80

    def __init__(
        self,
        model_path: Optional[str] = None,
        threads: int = 2,
    ) -> None:
        import onnxruntime as ort

        path = model_path or self._download_model()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._sess = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )

        # Auto-detect input/output node names and expected input shape
        inp = self._sess.get_inputs()[0]
        self._in_name: str = inp.name
        self._in_shape: list = inp.shape
        self._out_name: str = self._sess.get_outputs()[0].name

        log(logger, logging.INFO, "WeSpeaker ONNX loaded",
            model=str(path), in_name=self._in_name,
            in_shape=self._in_shape, out_name=self._out_name)

    @classmethod
    def _download_model(cls) -> str:
        """Download the ONNX model from HuggingFace Hub and return local path."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download the WeSpeaker ONNX model. "
                "Install it with: pip install huggingface_hub"
            ) from e

        log(logger, logging.INFO, "downloading WeSpeaker ONNX from HuggingFace Hub",
            repo=cls.HF_REPO, file=cls.HF_FILE,
            hint="this only happens once; model is cached locally")
        try:
            path = hf_hub_download(cls.HF_REPO, cls.HF_FILE)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download WeSpeaker ONNX model from "
                f"'{cls.HF_REPO}/{cls.HF_FILE}'.  "
                f"Check your internet connection and huggingface_hub version.\n"
                f"Original error: {e}"
            ) from e
        return path

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return L2-normalised 256-dim speaker embedding."""
        audio = np.asarray(audio, dtype=np.float32).ravel()

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            from voice_firewall.audio.dsp import resample_to
            audio = resample_to(audio, sample_rate, 16000)

        # Build Fbank features (T, 80)
        feats = extract_fbank(audio, sr=16000, n_mels=self._FBANK_DIM)

        # Add batch dimension: (1, T, 80)
        inp = feats[np.newaxis, :, :].astype(np.float32)

        # Determine if the model wants (B, T, F) or (B, F, T) by checking shape
        # The onnx-community model uses (B, T, F) = (1, T, 80)
        in_shape = self._in_shape
        if len(in_shape) >= 3:
            last_dim = in_shape[-1]
            if isinstance(last_dim, int) and last_dim == self._FBANK_DIM:
                pass  # already (1, T, 80) — correct
            elif isinstance(in_shape[-2], int) and in_shape[-2] == self._FBANK_DIM:
                # model expects (1, 80, T) → transpose
                inp = inp.transpose(0, 2, 1)

        emb = self._sess.run([self._out_name], {self._in_name: inp})[0]
        emb = emb.reshape(-1).astype(np.float32)
        return _l2norm(emb)

    @property
    def dim(self) -> int:
        return self._EMBED_DIM


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_embedder(
    model_name: str = "wespeaker-resnet34",
    model_path: Optional[str] = None,
    threads: int = 2,
    offline: bool = False,
) -> SpeakerEmbedder:
    """Return a ready-to-use embedder.

    Parameters
    ----------
    model_name : ``"wespeaker-resnet34"`` (default) or ``"stub"``.
    model_path : explicit local path to a ``.onnx`` file (skips download).
    threads    : ONNX thread count.
    offline    : if True and model_path is None → return StubEmbedder and log warning.
    """
    if model_name == "stub":
        return StubEmbedder()

    if offline and model_path is None:
        log(logger, logging.WARNING,
            "offline mode + no model_path → using StubEmbedder (no speaker verification)")
        return StubEmbedder()

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        log(logger, logging.WARNING,
            "onnxruntime not installed → using StubEmbedder (no speaker verification)",
            hint="pip install onnxruntime")
        return StubEmbedder()

    try:
        return OnnxWeSpeakerEmbedder(model_path=model_path, threads=threads)
    except Exception as exc:
        log(logger, logging.ERROR,
            "failed to load WeSpeaker ONNX embedder → falling back to StubEmbedder",
            err=str(exc))
        return StubEmbedder()
