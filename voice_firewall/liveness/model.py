"""Anti-spoofing / liveness model interface (Phase 4).

Design rules (non-negotiable):
1. This guard NEVER drops audio on its own.  It is a WARNING signal only.
2. It is NEVER the sole proof of identity.  G3 (speaker) and G6 (action) are
   the gates.  Liveness is an escalation hint.
3. The README honest-limits section MUST NOT be edited to claim this stops
   deepfakes.  A perfect voice clone will score near 1.0.

Interface
---------
Any model that implements ``AntispoofModel.score(audio) -> float`` can be
plugged in.  Built-in options:

``StubAntispoofModel``  — returns 0.5 (unknown); used when no model is loaded.
``OnnxAntispoofModel``  — loads an ONNX model from a local path.
                           Input:  float32 array of shape (1, N) at 16 kHz.
                           Output: float32 scalar liveness score in [0, 1].
                           0 = likely spoof; 1 = likely live.
``build_model(cfg)``    — factory; returns OnnxAntispoofModel when available,
                           else StubAntispoofModel with a one-time warning.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AntispoofModel(ABC):
    """Abstract anti-spoof / liveness scorer."""

    @abstractmethod
    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Return a liveness score in [0, 1].

        0.0 = almost certainly synthetic / spoofed.
        1.0 = almost certainly a live human speaker.
        0.5 = unknown / model not loaded.

        IMPORTANT: This is a probabilistic estimate. It MUST NOT be used as
        the sole gate for any action. A perfect clone may score near 1.0.
        """

    @property
    def available(self) -> bool:
        """True if a real model is loaded (not a stub)."""
        return False


# ---------------------------------------------------------------------------
# Stub — always returns 0.5 (unknown)
# ---------------------------------------------------------------------------

class StubAntispoofModel(AntispoofModel):
    """Stub used when no ONNX anti-spoof model is configured.

    Returns 0.5 (unknown) so downstream logic treats it as no information.
    """

    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        return 0.5

    @property
    def available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# ONNX anti-spoof model
# ---------------------------------------------------------------------------

class OnnxAntispoofModel(AntispoofModel):
    """Loads an ONNX anti-spoof model from a local file.

    Expected ONNX contract
    ----------------------
    - Input  : float32 of shape (1, T)   where T = number of samples at 16 kHz
    - Output : float32 scalar or (1,) or (1, 2) [bonafide_prob, spoof_prob]

    If the model outputs two values, the first is interpreted as the bonafide
    (live) probability.  Single-output models are used directly.

    Compatible with AASIST-ONNX export and similar architectures.
    """

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort  # noqa: PLC0415
        self._path = Path(model_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("OnnxAntispoofModel loaded: %s", self._path.name)

    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> float:
        """Run the model and return a liveness score in [0, 1]."""
        x = audio.astype(np.float32).reshape(1, -1)
        out = self._session.run(None, {self._input_name: x})[0]
        out = np.asarray(out).ravel()
        if out.size >= 2:
            # Binary classifier: [bonafide, spoof] or [spoof, bonafide]
            # Convention: first output = bonafide (live) probability.
            return float(np.clip(out[0], 0.0, 1.0))
        return float(np.clip(out[0], 0.0, 1.0))

    @property
    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_path: Optional[str]) -> AntispoofModel:
    """Build a liveness model from a path, or a stub if unavailable.

    Parameters
    ----------
    model_path: Path to an ONNX model file, or None / empty string.

    Returns
    -------
    OnnxAntispoofModel if the file exists and onnxruntime is installed,
    StubAntispoofModel otherwise (with a one-time WARNING log).
    """
    if not model_path:
        logger.warning(
            "LiveVoiceGuard: no model path configured — using StubAntispoofModel. "
            "Liveness score will always be 0.5 (unknown). "
            "Configure guards.live_voice.model_path to enable real anti-spoofing."
        )
        return StubAntispoofModel()

    p = Path(model_path)
    if not p.exists():
        logger.warning(
            "LiveVoiceGuard: model file not found at '%s' — using StubAntispoofModel.",
            model_path,
        )
        return StubAntispoofModel()

    try:
        return OnnxAntispoofModel(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LiveVoiceGuard: failed to load ONNX model '%s': %s — using stub.",
            model_path, exc,
        )
        return StubAntispoofModel()
