"""Deepgram STT adapter — dual-transcription backend (Phase 4).

Implements ``DualTranscriber`` using the Deepgram REST API with two different
model tiers to produce independent transcripts for comparison.

Install::
    pip install deepgram-sdk

Usage::
    from voice_firewall.adapters.deepgram import DeepgramDualTranscriber
    transcriber = DeepgramDualTranscriber(api_key="dg-...")
    # pass to WordsGuard(dual_transcriber=transcriber)

Security: the API key is passed at construction time and NEVER logged.
"""
from __future__ import annotations

import io
import logging
import wave
from typing import Optional

import numpy as np

from voice_firewall.words.dual_transcribe import DualTranscriber

logger = logging.getLogger(__name__)


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class DeepgramDualTranscriber(DualTranscriber):
    """Transcribes audio twice via the Deepgram API (nova-2 vs enhanced models).

    Run A: model="nova-2"    (highest accuracy)
    Run B: model="enhanced"  (independent architecture)

    Parameters
    ----------
    api_key: Deepgram API key — loaded from ``DEEPGRAM_API_KEY`` env var if omitted.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        import os  # noqa: PLC0415
        key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            logger.warning(
                "DeepgramDualTranscriber: no API key provided. "
                "Set DEEPGRAM_API_KEY or pass api_key=... to the constructor."
            )
            self._client = None
        else:
            try:
                from deepgram import DeepgramClient  # noqa: PLC0415
                self._client = DeepgramClient(key)
                self._available = True
            except ImportError:
                logger.warning(
                    "DeepgramDualTranscriber: deepgram-sdk not installed. "
                    "Install with: pip install deepgram-sdk"
                )
                self._client = None

    def transcribe_pair(self, audio: np.ndarray) -> tuple[str, str]:
        if self._client is None:
            return "", ""
        wav_bytes = _to_wav_bytes(audio)
        a = self._call(wav_bytes, model="nova-2")
        b = self._call(wav_bytes, model="enhanced")
        return a, b

    def _call(self, wav_bytes: bytes, model: str) -> str:
        try:
            from deepgram import PrerecordedOptions, FileSource  # noqa: PLC0415
            source: FileSource = {"buffer": wav_bytes, "mimetype": "audio/wav"}
            opts = PrerecordedOptions(model=model, language="en")
            resp = self._client.listen.prerecorded.v("1").transcribe_file(source, opts)
            return resp.results.channels[0].alternatives[0].transcript
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepgramDualTranscriber: API call failed: %s", exc)
            return ""
