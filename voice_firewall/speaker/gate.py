"""Rolling cosine-similarity gate for speaker verification (Phase 1 Step 4).

Design
------
- Two thresholds: ``threshold`` (normal) and ``strict_threshold`` (while assistant
  is speaking — must be >= threshold, never looser). Step 5 enforces this invariant.
- ``observe_mode=True``: scores are logged but the gate never drops.
- Profile is an L2-normalised centroid of enrollment embeddings (float32 numpy).
- ``RollingEmbeddingGate`` accumulates embeddings across the last N segments and
  scores the centroid so short noisy segments don't flip the gate.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from voice_firewall._logging import log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    accepted: bool          # True → segment may proceed
    score: float            # cosine similarity vs enrolled profile centroid
    threshold_used: float   # which threshold was applied this call
    observe_mode: bool      # True → gate is advisory only, never drops
    reason: str = ""


# ---------------------------------------------------------------------------
# Simple one-shot gate (single segment)
# ---------------------------------------------------------------------------

class CosineSpeakerGate:
    """Compare an embedding against the enrolled profile centroid.

    For short segments or noisy conditions prefer ``RollingEmbeddingGate``.
    """

    def __init__(
        self,
        profile: np.ndarray,
        threshold: float,
        strict_threshold: float,
        observe_mode: bool = False,
    ) -> None:
        norm = np.linalg.norm(profile)
        self.profile = profile / (norm + 1e-12)
        self.threshold = threshold
        # Strict threshold (while assistant speaking) must NEVER be looser.
        self.strict_threshold = max(strict_threshold, threshold)
        self.observe_mode = observe_mode

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    def check(self, embedding: np.ndarray, assistant_speaking: bool = False) -> GateResult:
        threshold = self.strict_threshold if assistant_speaking else self.threshold
        score = self.cosine(embedding, self.profile)
        accepted = self.observe_mode or score >= threshold

        reason = (
            "observe" if self.observe_mode
            else ("pass" if accepted else "low-score")
        )
        return GateResult(
            accepted=accepted,
            score=score,
            threshold_used=threshold,
            observe_mode=self.observe_mode,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Rolling embedding gate (accumulates segments → centroid score)
# ---------------------------------------------------------------------------

class RollingEmbeddingGate:
    """Score the centroid of the last ``window`` embeddings vs the profile.

    Smooths over short noisy segments; resets after ``max_gap_s`` of silence.
    """

    def __init__(
        self,
        profile: np.ndarray,
        threshold: float,
        strict_threshold: float,
        observe_mode: bool = False,
        window: int = 5,
    ) -> None:
        norm = np.linalg.norm(profile)
        self._profile = profile / (norm + 1e-12)
        self._threshold = threshold
        self._strict_threshold = max(strict_threshold, threshold)
        self._observe = observe_mode
        self._window = window
        self._recent: deque[np.ndarray] = deque(maxlen=window)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def strict_threshold(self) -> float:
        return self._strict_threshold

    @property
    def observe_mode(self) -> bool:
        return self._observe

    def reset(self) -> None:
        self._recent.clear()

    def push(self, embedding: np.ndarray, assistant_speaking: bool = False) -> GateResult:
        """Add ``embedding`` and return the current gate decision."""
        emb = embedding.astype(np.float64).ravel()
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        self._recent.append(emb.astype(np.float32))

        # Centroid of recent embeddings
        centroid = np.mean(np.stack(list(self._recent)), axis=0).astype(np.float64)
        c_norm = np.linalg.norm(centroid)
        centroid = centroid / (c_norm + 1e-12)

        profile = self._profile.astype(np.float64)
        score = float(np.dot(centroid, profile))

        threshold = self._strict_threshold if assistant_speaking else self._threshold
        accepted = self._observe or score >= threshold

        reason = "observe" if self._observe else ("pass" if accepted else "low-score")
        return GateResult(
            accepted=accepted,
            score=score,
            threshold_used=threshold,
            observe_mode=self._observe,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Profile I/O
# ---------------------------------------------------------------------------

def load_profile(path: str) -> Optional[np.ndarray]:
    """Load a voice profile from a .npy file.  Returns None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    arr = np.load(str(p))
    norm = np.linalg.norm(arr)
    return (arr / (norm + 1e-12)).astype(np.float32)


def save_profile(embeddings: list[np.ndarray], path: str) -> np.ndarray:
    """Average embeddings into a centroid, L2-normalise, and save.

    Parameters
    ----------
    embeddings : list of L2-normalised embedding vectors from enrollment phrases.
    path       : file path to save the .npy profile (parent dirs are created).
    """
    if not embeddings:
        raise ValueError("No embeddings provided for enrollment")
    centroid = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    centroid = centroid / (norm + 1e-12)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), centroid)
    log(logger, logging.INFO, "speaker profile saved",
        path=str(p), n_phrases=len(embeddings))
    return centroid
