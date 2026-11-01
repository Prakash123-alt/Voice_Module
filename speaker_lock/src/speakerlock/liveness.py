"""Heuristic liveness estimate: is this window a live voice at the mic or a replayed recording?

This is NOT a certified anti-spoofing model (an AASIST-class detector is a separate model and
dependency). It scores three cheap channel-quality cues that playback through a loudspeaker +
room + mic chain typically degrades, and returns a 0..1 estimate (1 = looks like a live voice):

  * spectral flatness   - replayed speech is filtered by the speaker+room response, which tends
                          to flatten its fine spectral structure vs a direct voice
  * high-frequency band - laptop/phone speakers roll off above ~4 kHz; live speech keeps more
  * clipping            - over-driven playback chains hard-clip; near-clipped samples count

Weights are calibrated conservatively: the score is informative metadata (a "liveness token")
for downstream policy, not a gate on its own. Combine with `gate.invert`/`strict_turns` and a
real anti-spoof model for adversarial deployments.
"""
from __future__ import annotations

import numpy as np

from .config import SAMPLE_RATE


def liveness_score(audio: np.ndarray) -> float:
    """Return a 0..1 liveness estimate for a speech window (float32 16 kHz)."""
    x = np.asarray(audio, np.float32).ravel()
    if x.shape[0] < SAMPLE_RATE // 4 or float(np.max(np.abs(x))) < 1e-4:
        return 0.5                              # too little evidence: neutral
    spec = np.abs(np.fft.rfft(x)) + 1e-12
    flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))          # 0..1, higher = flatter
    hi_bin = int(4000 / (SAMPLE_RATE / 2) * spec.shape[0])
    hi_ratio = float(np.sum(spec[hi_bin:]) / np.sum(spec))               # energy above 4 kHz
    clip_frac = float(np.mean(np.abs(x) > 0.98))

    flat_score = 1.0 - min(flat / 0.10, 1.0)      # voiced speech sits well under ~0.05 flatness
    hi_score = min(hi_ratio / 0.02, 1.0)          # ~2% of spectral energy above 4 kHz = live-ish
    clip_score = 1.0 - min(clip_frac * 20.0, 1.0)
    return round(0.45 * flat_score + 0.40 * hi_score + 0.15 * clip_score, 3)
