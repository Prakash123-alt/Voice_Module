"""Text-to-speech. Mode A: audio output modality via OpenRouter chat completions.
Mode B: local Piper. Both yield float32 @ 16 kHz chunks."""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from .dsp import from_wav_bytes, resample, to_float32
from .interfaces import TTSEngine
from .logging_utils import log
from .openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


class OpenRouterTTS(TTSEngine):
    """Asks an audio-output model to read the text verbatim. Returns one chunk per call."""

    def __init__(self, client: OpenRouterClient, model: str, voice: str = "alloy", sample_rate: int = 16000) -> None:
        self.client, self.model, self.voice, self.rate = client, model, voice, sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        messages = [{"role": "system", "content": "Read the user's text aloud exactly as written. Do not add anything."},
                    {"role": "user", "content": text}]
        resp = await self.client.chat(messages, self.model, modalities=["text", "audio"],
                                      audio={"voice": self.voice, "format": "wav"})
        audio_obj = (resp.get("choices") or [{}])[0].get("message", {}).get("audio") or {}
        data = audio_obj.get("data")
        if not data:
            log(logger, logging.WARNING, "no audio in OpenRouter TTS response", model=self.model)
            return
        raw = base64.b64decode(data)
        if raw[:4] == b"RIFF":
            pcm, rate = from_wav_bytes(raw)
        else:  # pcm16 @ 24 kHz is the usual fallback
            pcm, rate = to_float32(np.frombuffer(raw, dtype=np.int16)), 24000
        yield resample(pcm, rate, self.rate)


class PiperTTS(TTSEngine):
    def __init__(self, model_path: str, speaker: Optional[int] = None, sample_rate: int = 16000) -> None:
        from piper import PiperVoice
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Piper voice {p} not found - see README for download instructions")
        self._voice = PiperVoice.load(str(p))
        self._speaker = speaker
        self.rate = sample_rate
        self._src_rate = int(self._voice.config.sample_rate)
        for _ in self._chunks("Ready."):  # warm-up: the first ONNX run is ~20x slower
            pass
        log(logger, logging.INFO, "piper ready", model=p.name, rate=self._src_rate)

    def _chunks(self, text: str):
        """Piper >= 1.3 exposes synthesize() yielding AudioChunk; older versions synthesize_stream_raw()."""
        if hasattr(self._voice, "synthesize_stream_raw"):
            for raw in self._voice.synthesize_stream_raw(text, speaker_id=self._speaker):
                yield to_float32(np.frombuffer(raw, dtype=np.int16))
            return
        try:
            from piper import SynthesisConfig
            cfg = SynthesisConfig(speaker_id=self._speaker) if self._speaker is not None else None
            it = self._voice.synthesize(text, syn_config=cfg) if cfg else self._voice.synthesize(text)
        except ImportError:
            it = self._voice.synthesize(text)
        for chunk in it:
            if hasattr(chunk, "audio_float_array"):
                yield np.asarray(chunk.audio_float_array, dtype=np.float32)
            else:
                yield to_float32(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = False

        def worker() -> None:
            try:
                for c in self._chunks(text):
                    if stop:
                        break
                    loop.call_soon_threadsafe(q.put_nowait, resample(c, self._src_rate, self.rate))
            except Exception as e:  # surfaced to the consumer
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, worker)
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop = True  # worker thread exits at its next chunk boundary


def build_tts(cfg, client: Optional[OpenRouterClient], sample_rate: int) -> TTSEngine:
    if cfg.mode == "openrouter":
        if client is None:
            raise RuntimeError("tts.mode=openrouter requires an OpenRouter client")
        return OpenRouterTTS(client, cfg.openrouter_model, cfg.openrouter_voice, sample_rate)
    return PiperTTS(cfg.piper_model, cfg.piper_speaker, sample_rate)
