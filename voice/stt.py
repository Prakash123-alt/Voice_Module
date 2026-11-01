"""Speech-to-text. Mode A: audio-capable chat model via OpenRouter (`input_audio`).
Mode B: local faster-whisper. Either way the only network traffic is OpenRouter."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Optional

import numpy as np

from .dsp import to_wav_bytes
from .interfaces import STTEngine
from .logging_utils import log
from .openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

_TRANSCRIBE_PROMPT = ("Transcribe the speech in this audio verbatim. Output only the transcript text, "
                      "with no quotes, labels or commentary. If there is no intelligible speech, output nothing.")


class OpenRouterSTT(STTEngine):
    def __init__(self, client: OpenRouterClient, model: str, language: str = "en", sample_rate: int = 16000) -> None:
        self.client, self.model, self.language, self.rate = client, model, language, sample_rate

    async def transcribe(self, audio: np.ndarray) -> str:
        b64 = base64.b64encode(to_wav_bytes(audio, self.rate)).decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": _TRANSCRIBE_PROMPT + (f" Language: {self.language}." if self.language else "")},
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]}]
        resp = await self.client.chat(messages, self.model, temperature=0.0, max_tokens=400)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if isinstance(content, list):  # some providers return content parts
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        return content.strip().strip('"')


class WhisperSTT(STTEngine):
    def __init__(self, model_size: str = "base.en", compute_type: str = "int8", language: Optional[str] = "en",
                 beam_size: int = 1, initial_prompt: str = "", hotwords: str = "") -> None:
        import torch
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and compute_type == "int8":
            compute_type = "float16"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                   cpu_threads=os.cpu_count() or 4)
        self.language = language
        self.beam_size = max(1, beam_size)
        self.initial_prompt = initial_prompt.strip() or None
        self.hotwords = hotwords.strip() or None
        self._run(np.zeros(8000, dtype=np.float32))  # warm-up: first inference is several times slower
        log(logger, logging.INFO, "faster-whisper ready", model=model_size, device=device, compute=compute_type,
            beam=self.beam_size, prompt=bool(self.initial_prompt), hotwords=bool(self.hotwords))

    def _run(self, audio: np.ndarray) -> str:
        # utterances are already VAD-segmented, so whisper's own VAD and timestamps are skipped
        segments, _ = self._model.transcribe(audio.astype(np.float32), language=self.language,
                                             beam_size=self.beam_size, initial_prompt=self.initial_prompt,
                                             hotwords=self.hotwords, vad_filter=False,
                                             condition_on_previous_text=False, without_timestamps=True)
        return _segments_text(segments)

    async def transcribe(self, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self._run, audio)


def _segments_text(segments) -> str:
    """Join segment texts, dropping likely hallucinations. Whisper occasionally invents repetitive
    text ("Low, low, low...") for unclear audio; its own confidence stats catch most of these."""
    kept = []
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        if (s.avg_logprob < -1.0 or s.compression_ratio > 2.4
                or (s.no_speech_prob > 0.6 and s.avg_logprob < -0.5)):
            log(logger, logging.INFO, "stt segment dropped",
                text=text, logprob=round(s.avg_logprob, 2),
                compression=round(s.compression_ratio, 2), no_speech=round(s.no_speech_prob, 2))
            continue
        kept.append(text)
    return " ".join(kept).strip()


def build_stt(cfg, client: Optional[OpenRouterClient], sample_rate: int) -> STTEngine:
    if cfg.mode == "openrouter":
        if client is None:
            raise RuntimeError("stt.mode=openrouter requires an OpenRouter client")
        return OpenRouterSTT(client, cfg.openrouter_model, cfg.language, sample_rate)
    return WhisperSTT(cfg.local_model, cfg.local_compute_type, cfg.language or None,
                      cfg.beam_size, cfg.initial_prompt, cfg.hotwords)
