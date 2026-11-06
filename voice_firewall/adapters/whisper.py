"""Whisper STT adapter — dual-transcription backend (Phase 4).

Implements ``DualTranscriber`` using faster-whisper (or any Whisper-compatible
backend) with two different beam settings to produce diverging transcripts for
comparison.

Install::
    pip install faster-whisper

Usage::
    from voice_firewall.adapters.whisper import WhisperDualTranscriber
    from voice_firewall._config import WordsConfig
    from voice_firewall.guards.g5_words import WordsGuard

    transcriber = WhisperDualTranscriber(model_size="tiny")
    guard = WordsGuard(WordsConfig(dual_transcribe_on_high_risk=True),
                       dual_transcriber=transcriber)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from voice_firewall.words.dual_transcribe import DualTranscriber

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WhisperDualTranscriber(DualTranscriber):
    """Transcribes audio twice with different Whisper beam sizes.

    Run A: beam_size=5, temperature=0  (greedy-ish)
    Run B: beam_size=1, temperature=0.4 (stochastic)

    Significant divergence between A and B raises the
    ``dual-transcribe-mismatch`` flag in G5.

    Parameters
    ----------
    model_size: Whisper model size string passed to ``faster_whisper.WhisperModel``.
                "tiny" is fast but less accurate; "base" is recommended.
    device:     "cpu" or "cuda".
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
            self._model = WhisperModel(model_size, device=device, compute_type="int8")
            self._available = True
            logger.info("WhisperDualTranscriber loaded: %s on %s", model_size, device)
        except ImportError:
            logger.warning(
                "WhisperDualTranscriber: faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
            self._model = None
            self._available = False

    def transcribe_pair(self, audio: np.ndarray) -> tuple[str, str]:
        if not self._available or self._model is None:
            return "", ""
        a = self._transcribe(audio, beam_size=5, temperature=0.0)
        b = self._transcribe(audio, beam_size=1, temperature=0.4)
        return a, b

    def _transcribe(self, audio: np.ndarray, beam_size: int, temperature: float) -> str:
        segments, _ = self._model.transcribe(
            audio,
            beam_size=beam_size,
            temperature=temperature,
            language="en",
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments)
