"""Speaker verification: embed a speech segment, cosine-compare with the enrolled profile."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .interfaces import SpeakerVerifier, VerificationResult
from .logging_utils import log

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def average_embedding(embs: list[np.ndarray]) -> np.ndarray:
    """L2-normalise each embedding, average, re-normalise (standard centroid)."""
    m = np.stack([e / (np.linalg.norm(e) + 1e-12) for e in embs]).mean(axis=0)
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


class Embedder(ABC):
    @abstractmethod
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """float32 16 kHz mono -> 1-D embedding."""


class ResemblyzerEmbedder(Embedder):
    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        from resemblyzer import VoiceEncoder
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._enc = VoiceEncoder(device=dev, verbose=False)
        self.embed(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1)  # warm-up (~400 ms cold)
        log(logger, logging.INFO, "speaker embedder ready", backend="resemblyzer", device=dev)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio.astype(np.float32), source_sr=16000)
        if wav.shape[0] < 1600:  # <100 ms after trimming; pad so encoder has at least one partial
            wav = np.pad(wav, (0, 1600 - wav.shape[0]))
        return self._enc.embed_utterance(wav).astype(np.float32)


class ECAPAEmbedder(Embedder):
    """ECAPA-TDNN via SpeechBrain (speechbrain/spkrec-ecapa-voxceleb)."""

    def __init__(self, device: Optional[str] = None) -> None:
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="models/ecapa", run_opts={"device": dev})
        log(logger, logging.INFO, "speaker embedder ready", backend="ecapa", device=dev)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            t = self._torch.from_numpy(audio.astype(np.float32))[None, :]
            return self._enc.encode_batch(t).squeeze().cpu().numpy().astype(np.float32)


def build_embedder(backend: str) -> Embedder:
    if backend == "ecapa":
        return ECAPAEmbedder()
    if backend == "resemblyzer":
        return ResemblyzerEmbedder()
    raise ValueError(f"unknown speaker backend {backend!r}")


class ProfileSpeakerVerifier(SpeakerVerifier):
    def __init__(self, embedder: Embedder, profile: np.ndarray, threshold: float = 0.65,
                 min_segment_ms: int = 400, sample_rate: int = 16000) -> None:
        self.embedder, self.profile, self.threshold = embedder, np.asarray(profile, dtype=np.float32), threshold
        self.min_samples = int(min_segment_ms * sample_rate / 1000)

    @classmethod
    def from_file(cls, embedder: Embedder, path: str | Path, **kw) -> "ProfileSpeakerVerifier":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"speaker profile {p} missing - run `python enroll.py` first")
        return cls(embedder, np.load(p), **kw)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        return self.embedder.embed(audio)

    def verify(self, audio: np.ndarray, min_samples: Optional[int] = None) -> VerificationResult:
        if audio.shape[0] < (self.min_samples if min_samples is None else min_samples):
            return VerificationResult(False, 0.0, self.threshold)
        score = cosine_similarity(self.embed(audio), self.profile)
        return VerificationResult(score >= self.threshold, score, self.threshold)


def suggest_threshold(genuine: list[float], impostor: list[float] | None = None, margin: float = 0.15) -> dict:
    """Threshold suggestion. With only genuine scores: mean - 2.5*std, floored by margin below the
    minimum. Enrollment scores come from one quiet sitting, so real-world scores vary more; the
    margin is deliberately generous. With impostor scores too: the midpoint that maximises
    separation (EER-style)."""
    g = np.asarray(genuine, dtype=np.float64)
    out = {"genuine_mean": float(g.mean()), "genuine_std": float(g.std()), "genuine_min": float(g.min())}
    if impostor:
        i = np.asarray(impostor, dtype=np.float64)
        cands = np.linspace(min(i.min(), g.min()), max(i.max(), g.max()), 400)
        far = np.array([(i >= c).mean() for c in cands])
        frr = np.array([(g < c).mean() for c in cands])
        k = int(np.argmin(np.abs(far - frr)))
        out.update(impostor_mean=float(i.mean()), impostor_max=float(i.max()), eer=float((far[k] + frr[k]) / 2),
                   suggested=float(cands[k]))
    else:
        out["suggested"] = float(max(0.3, min(g.mean() - 2.5 * g.std(), g.min() - margin)))
    out["suggested"] = round(out["suggested"], 3)
    return out
