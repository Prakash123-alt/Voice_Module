"""Enrollment: a ~5 s reference clip -> extractor conditioning + speaker d-vector, stored per user ID.

The d-vector is computed the same way the runtime verifier sees audio: the clip is first passed through the
extractor (conditioned on itself) and only VAD-positive frames are embedded. Enrolling in the same domain
as verification avoids the systematic score drop you get when enrollment audio is raw but runtime audio is
processed.

Privacy: `cond_enroll_mag` is a magnitude spectrogram of the reference clip, i.e. biometric data. Protect
the store directory accordingly; raw audio itself is never stored.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE
from .interfaces import VAD, Embedder, Extractor, TargetCondition, l2norm

MIN_SECONDS, RECOMMENDED_SECONDS = 3.0, 5.0
_USER_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class EnrollmentError(ValueError):
    pass


@dataclass
class Enrollment:
    user_id: str
    dvector: np.ndarray
    condition: TargetCondition
    meta: dict = field(default_factory=dict)


def check_clip(clip: np.ndarray) -> list[str]:
    """Raise EnrollmentError for unusable clips; return warnings for marginal ones."""
    clip = np.asarray(clip, np.float32)
    dur = clip.shape[0] / SAMPLE_RATE
    if dur < MIN_SECONDS:
        raise EnrollmentError(f"clip is {dur:.1f} s; need at least {MIN_SECONDS:.0f} s ({RECOMMENDED_SECONDS:.0f} s recommended)")
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(clip ** 2))) + 1e-10)
    if rms_db < -50:
        raise EnrollmentError(f"clip is too quiet ({rms_db:.0f} dBFS) - check the microphone")
    warnings = []
    if float(np.mean(np.abs(clip) > 0.99)) > 0.001:
        warnings.append("clip is clipping - move back from the microphone or lower the gain")
    if dur < RECOMMENDED_SECONDS - 0.5:
        warnings.append(f"clip is {dur:.1f} s; {RECOMMENDED_SECONDS:.0f} s gives a more reliable profile")
    return warnings


def speech_only(x: np.ndarray, vad: VAD, threshold: float = 0.5) -> np.ndarray:
    vad.reset()
    n = vad.frame_samples
    frames = [x[i:i + n] for i in range(0, x.shape[0] - n + 1, n)]
    voiced = [f for f in frames if vad.probability(f) >= threshold]
    vad.reset()
    return np.concatenate(voiced) if voiced else np.zeros(0, np.float32)


def enroll(user_id: str, clip: np.ndarray, extractor: Extractor, embedder: Embedder,
           vad: Optional[VAD] = None, vad_threshold: float = 0.5) -> Enrollment:
    if not _USER_ID.match(user_id):
        raise EnrollmentError("user_id must be 1-64 chars of letters, digits, '_', '-', '.'")
    clip = np.asarray(clip, np.float32)
    warnings = check_clip(clip)
    cond = extractor.condition(clip)
    extractor.start(cond)
    isolated = np.concatenate([extractor.process(clip), extractor.flush()])
    voiced = speech_only(isolated, vad, vad_threshold) if vad is not None else isolated
    if voiced.shape[0] < 1.5 * SAMPLE_RATE:
        raise EnrollmentError(f"only {voiced.shape[0] / SAMPLE_RATE:.1f} s of speech detected; speak continuously for ~5 s")
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "clip_s": round(clip.shape[0] / SAMPLE_RATE, 2),
            "speech_s": round(voiced.shape[0] / SAMPLE_RATE, 2), "embedder": type(embedder).__name__,
            "extractor": type(extractor).__name__, "warnings": warnings}
    return Enrollment(user_id, l2norm(embedder.embed(voiced)), cond, meta)


class UserStore:
    """One `<user_id>.npz` per enrolled user (d-vector + extractor conditioning + JSON metadata)."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        if not _USER_ID.match(user_id):
            raise EnrollmentError(f"invalid user_id {user_id!r}")
        return self.dir / f"{user_id}.npz"

    def save(self, e: Enrollment) -> Path:
        p = self._path(e.user_id)
        tmp = p.with_suffix(".tmp.npz")
        arrays = {f"cond_{k}": v for k, v in e.condition.arrays.items()}
        np.savez(tmp, dvector=e.dvector, meta=np.array(json.dumps(e.meta)), **arrays)
        tmp.replace(p)
        return p

    def load(self, user_id: str) -> Enrollment:
        p = self._path(user_id)
        if not p.exists():
            raise KeyError(f"user {user_id!r} is not enrolled (looked in {self.dir})")
        with np.load(p, allow_pickle=False) as z:
            cond = TargetCondition({k[5:]: z[k] for k in z.files if k.startswith("cond_")})
            return Enrollment(user_id, z["dvector"], cond, json.loads(str(z["meta"])))

    def users(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.npz") if not p.stem.endswith(".tmp"))

    def profiles(self) -> dict[str, np.ndarray]:
        return {u: self.load(u).dvector for u in self.users()}

    def delete(self, user_id: str) -> None:
        self._path(user_id).unlink(missing_ok=True)
