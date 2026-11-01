"""Clean stream -> faster-whisper. Verified speech is accumulated into utterances, which are transcribed
when the speaker pauses (or the gate closes). Only audio that passed the gate ever reaches Whisper."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

from ..config import SAMPLE_RATE
from ..pipeline import ChunkMeta

Transcriber = Callable[[np.ndarray], str]


def faster_whisper_transcriber(model: str = "base.en", compute_type: str = "int8", language: Optional[str] = "en",
                               beam_size: int = 1, **kw) -> Transcriber:
    from faster_whisper import WhisperModel  # noqa: PLC0415
    m = WhisperModel(model, device="cpu", compute_type=compute_type)

    def run(audio: np.ndarray) -> str:
        segs, _ = m.transcribe(audio, language=language, beam_size=beam_size, vad_filter=False,
                               condition_on_previous_text=False, without_timestamps=True, **kw)
        return " ".join(s.text.strip() for s in segs).strip()
    return run


class WhisperAdapter:
    def __init__(self, transcriber: Optional[Transcriber] = None, end_silence_ms: int = 600,
                 min_utterance_ms: int = 300, max_utterance_s: float = 20.0, vad_threshold: float = 0.5) -> None:
        self.transcribe = transcriber or faster_whisper_transcriber()
        self.end_silence = int(end_silence_ms * SAMPLE_RATE / 1000)
        self.min_len = int(min_utterance_ms * SAMPLE_RATE / 1000)
        self.max_len = int(max_utterance_s * SAMPLE_RATE)
        self.vad_threshold = vad_threshold
        self._buf: list[np.ndarray] = []
        self._len = self._silence = 0

    def feed(self, pcm: np.ndarray, meta: ChunkMeta) -> list[str]:
        """Feed one SpeakerLock output chunk; returns any transcripts that completed."""
        active = meta.verified and meta.vad >= self.vad_threshold
        if active:
            self._buf.append(pcm)
            self._len += pcm.shape[0]
            self._silence = 0
        elif self._buf:
            self._buf.append(pcm)                 # keep short pauses inside the utterance
            self._len += pcm.shape[0]
            self._silence += pcm.shape[0]
        if self._buf and (self._silence >= self.end_silence or self._len >= self.max_len or
                          (not meta.verified and self._silence > 0)):
            return self._finish()
        return []

    def flush(self) -> list[str]:
        return self._finish() if self._buf else []

    def _finish(self) -> list[str]:
        audio = np.concatenate(self._buf)
        audio = audio[: audio.shape[0] - self._silence] if self._silence else audio
        self._buf, self._len, self._silence = [], 0, 0
        if audio.shape[0] < self.min_len:
            return []
        text = self.transcribe(audio.astype(np.float32))
        return [text] if text else []
