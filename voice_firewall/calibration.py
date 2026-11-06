"""Calibration math — FAR/FRR, EER, and threshold recommendations.

All functions are pure NumPy; no audio hardware or model downloads needed.
Used by ``scripts/calibrate.py`` and by tests.

Definitions
-----------
owner_scores   : cosine-similarity values for segments known to be the owner.
nonowner_scores: cosine-similarity values for segments known NOT to be the owner.

FAR(t) = P(score >= t | nonowner) — false-accept rate at threshold t.
FRR(t) = P(score <  t | owner)   — false-reject rate at threshold t.

EER  : the threshold where FAR ≈ FRR (equal error rate).
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from voice_firewall._logging import log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core FAR/FRR math
# ---------------------------------------------------------------------------

def compute_far_frr(
    owner_scores: np.ndarray,
    nonowner_scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (thresholds, FAR, FRR) arrays for a sweep of threshold values.

    Returns
    -------
    thresholds : 1-D array of candidate thresholds sorted ascending.
    far        : FAR at each threshold (non-owner accept rate).
    frr        : FRR at each threshold (owner reject rate).
    """
    owner = np.asarray(owner_scores, dtype=np.float64).ravel()
    nonowner = np.asarray(nonowner_scores, dtype=np.float64).ravel()

    all_scores = np.concatenate([owner, nonowner])
    if thresholds is None:
        lo, hi = all_scores.min() - 1e-4, all_scores.max() + 1e-4
        thresholds = np.linspace(lo, hi, n_points)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)

    # Vectorised computation using broadcasting.
    # owner[None, :] shape (1, N_owner); thresholds[:, None] shape (n_points, 1)
    n_owner = owner.size or 1
    n_nonowner = nonowner.size or 1

    frr = np.sum(owner[None, :] < thresholds[:, None], axis=1) / n_owner
    far = np.sum(nonowner[None, :] >= thresholds[:, None], axis=1) / n_nonowner

    return thresholds, far, frr


def find_eer(
    owner_scores: np.ndarray,
    nonowner_scores: np.ndarray,
    n_points: int = 10_000,
) -> tuple[float, float]:
    """Return (eer_threshold, eer_rate) — the threshold where FAR ≈ FRR.

    Uses interpolation to find the crossing point between FAR and FRR curves.
    """
    thresholds, far, frr = compute_far_frr(owner_scores, nonowner_scores, n_points=n_points)
    diff = far - frr
    # Find sign changes (crossings).
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        # No crossing → pick the threshold that minimises |FAR - FRR|.
        idx = int(np.argmin(np.abs(diff)))
        return float(thresholds[idx]), float((far[idx] + frr[idx]) / 2)
    # Use the first crossing and interpolate between the two adjacent points.
    i = int(sign_changes[0])
    # Linear interpolation between thresholds[i] and thresholds[i+1]
    d0, d1 = diff[i], diff[i + 1]
    t0, t1 = thresholds[i], thresholds[i + 1]
    if abs(d1 - d0) < 1e-12:
        t_eer = (t0 + t1) / 2.0
    else:
        t_eer = t0 - d0 * (t1 - t0) / (d1 - d0)
    # EER rate at the interpolated threshold
    f_far = np.interp(t_eer, thresholds, far)
    f_frr = np.interp(t_eer, thresholds, frr)
    eer_rate = (f_far + f_frr) / 2.0
    return float(t_eer), float(eer_rate)


def find_threshold_at_far(
    owner_scores: np.ndarray,
    nonowner_scores: np.ndarray,
    target_far: float,
    n_points: int = 10_000,
) -> Optional[float]:
    """Return the threshold where FAR <= target_far.

    Returns None if no threshold achieves the target FAR.
    """
    thresholds, far, _ = compute_far_frr(owner_scores, nonowner_scores, n_points=n_points)
    # We want the highest threshold (most permissive) where FAR <= target_far.
    # Because increasing the threshold lowers FAR (fewer non-owners accepted).
    candidates = thresholds[far <= target_far]
    if len(candidates) == 0:
        return None
    return float(candidates.min())  # lowest threshold that achieves target FAR


def frr_at_threshold(owner_scores: np.ndarray, threshold: float) -> float:
    """Return FRR at a specific threshold (owner reject rate)."""
    owner = np.asarray(owner_scores, dtype=np.float64).ravel()
    if owner.size == 0:
        return 0.0
    return float(np.mean(owner < threshold))


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Recommended thresholds and error-rate summary."""
    n_owner: int
    n_nonowner: int

    # EER point
    eer_threshold: float
    eer_rate: float                   # FAR = FRR = EER at this threshold

    # Conservative operation: FAR <= 1 %
    threshold_far1: Optional[float]   # None if 1% FAR is unachievable
    frr_at_far1: Optional[float]

    # Balanced operation: FAR <= 5 %
    threshold_far5: Optional[float]
    frr_at_far5: Optional[float]

    # Recommendation
    threshold: float                  # recommended normal threshold
    strict_threshold: float           # recommended strict (barge-in) threshold

    def report(self) -> str:
        """Human-readable calibration report."""
        lines = [
            "Voice Firewall — Calibration Report",
            "=" * 50,
            f"Samples:   {self.n_owner} owner  /  {self.n_nonowner} non-owner",
            "",
            f"EER        threshold={self.eer_threshold:.4f}  rate={self.eer_rate:.1%}",
        ]
        if self.threshold_far1 is not None:
            lines.append(
                f"FAR ≤ 1%   threshold={self.threshold_far1:.4f}"
                f"  FRR={self.frr_at_far1:.1%}"
            )
        else:
            lines.append("FAR ≤ 1%   not achievable with this dataset")

        if self.threshold_far5 is not None:
            lines.append(
                f"FAR ≤ 5%   threshold={self.threshold_far5:.4f}"
                f"  FRR={self.frr_at_far5:.1%}"
            )
        else:
            lines.append("FAR ≤ 5%   not achievable with this dataset")

        lines += [
            "",
            "RECOMMENDED THRESHOLDS",
            f"  threshold:        {self.threshold:.4f}  (normal speaker gate)",
            f"  strict_threshold: {self.strict_threshold:.4f}  (barge-in / assistant-speaking gate)",
            "",
            "To apply:  python scripts/calibrate.py --write",
        ]
        return "\n".join(lines)


def compute_calibration(
    owner_scores: np.ndarray,
    nonowner_scores: np.ndarray,
    strict_margin: float = 0.05,
) -> CalibrationResult:
    """Run full calibration and return a CalibrationResult.

    Parameters
    ----------
    owner_scores    : cosine-similarity values for owner utterances.
    nonowner_scores : cosine-similarity values for non-owner utterances.
    strict_margin   : extra margin added to ``threshold`` for ``strict_threshold``.
                      Clamped to [0.0, 1.0].
    """
    owner = np.asarray(owner_scores, dtype=np.float64).ravel()
    nonowner = np.asarray(nonowner_scores, dtype=np.float64).ravel()

    if owner.size == 0:
        raise ValueError("Need at least one owner score for calibration")
    if nonowner.size == 0:
        raise ValueError("Need at least one non-owner score for calibration")

    eer_t, eer_rate = find_eer(owner, nonowner)

    t_far1 = find_threshold_at_far(owner, nonowner, 0.01)
    frr1 = frr_at_threshold(owner, t_far1) if t_far1 is not None else None

    t_far5 = find_threshold_at_far(owner, nonowner, 0.05)
    frr5 = frr_at_threshold(owner, t_far5) if t_far5 is not None else None

    # Choose recommendation: prefer FAR ≤ 5% if available; fall back to EER.
    threshold = t_far5 if t_far5 is not None else eer_t
    strict_threshold = min(1.0, threshold + strict_margin)

    return CalibrationResult(
        n_owner=int(owner.size),
        n_nonowner=int(nonowner.size),
        eer_threshold=float(eer_t),
        eer_rate=float(eer_rate),
        threshold_far1=float(t_far1) if t_far1 is not None else None,
        frr_at_far1=float(frr1) if frr1 is not None else None,
        threshold_far5=float(t_far5) if t_far5 is not None else None,
        frr_at_far5=float(frr5) if frr5 is not None else None,
        threshold=float(threshold),
        strict_threshold=float(strict_threshold),
    )


# ---------------------------------------------------------------------------
# Score log (JSONL) — written by G3 in observe mode
# ---------------------------------------------------------------------------

class ScoreLog:
    """Thread-safe JSONL score log for observe-mode calibration data collection.

    Format (one JSON object per line):
        {"ts": 1234567890.123, "score": 0.85, "label": "unknown"}

    ``label`` is always ``"unknown"`` during live use.  For calibration,
    users run separate recordings and pass ``--owner``/``--nonowner`` directories
    to ``calibrate.py``, or manually label the log and pass ``--log``.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, score: float, label: str = "unknown") -> None:
        import time as _t
        entry = json.dumps({"ts": round(_t.time(), 3), "score": round(score, 6), "label": label})
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(entry + "\n")

    @classmethod
    def read(cls, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a labeled JSONL log and return (owner_scores, nonowner_scores).

        Only entries with ``label == "owner"`` or ``label == "nonowner"`` are used.
        Entries with ``label == "unknown"`` are ignored.
        """
        owner, nonowner = [], []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                lbl = entry.get("label", "unknown")
                score = float(entry["score"])
                if lbl == "owner":
                    owner.append(score)
                elif lbl == "nonowner":
                    nonowner.append(score)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return np.array(owner, dtype=np.float64), np.array(nonowner, dtype=np.float64)
