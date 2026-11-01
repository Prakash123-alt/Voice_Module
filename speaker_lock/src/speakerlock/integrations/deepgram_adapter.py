"""Clean stream -> Deepgram streaming STT (WebSocket, linear16 @ 16 kHz).

    async with DeepgramStreamer(os.environ["DEEPGRAM_API_KEY"], on_transcript=print) as dg:
        for pcm, meta in clean_chunks:
            await dg.send(pcm)

The stream SpeakerLock outputs is already zeroed whenever the gate is closed, so unverified speakers are
never transcribed, while Deepgram still receives continuous audio (no reconnects, stable endpointing).
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Optional
from urllib.parse import urlencode

import numpy as np

from ..audio_io import to_pcm16
from ..config import SAMPLE_RATE

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramStreamer:
    def __init__(self, api_key: str, on_transcript: Optional[Callable[[str, bool], None]] = None,
                 model: str = "nova-3", language: str = "en", interim_results: bool = True,
                 url: str = DEEPGRAM_URL, **params: str) -> None:
        if not api_key:
            raise ValueError("Deepgram API key is required (DEEPGRAM_API_KEY)")
        self._key, self._cb = api_key, on_transcript
        query = {"encoding": "linear16", "sample_rate": SAMPLE_RATE, "channels": 1, "model": model,
                 "language": language, "interim_results": str(interim_results).lower(), **params}
        self.url = f"{url}?{urlencode(query)}"
        self.finals: list[str] = []
        self._ws = None
        self._reader: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "DeepgramStreamer":
        from websockets.asyncio.client import connect  # noqa: PLC0415
        self._ws = await connect(self.url, additional_headers={"Authorization": f"Token {self._key}"})
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def send(self, pcm: np.ndarray) -> None:
        if self._ws is None:
            raise RuntimeError("not connected - use `async with DeepgramStreamer(...)`")
        if pcm.shape[0]:
            await self._ws.send(to_pcm16(pcm))

    async def _read(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type", "Results") != "Results":
                continue
            alts = msg.get("channel", {}).get("alternatives") or [{}]
            text = (alts[0].get("transcript") or "").strip()
            final = bool(msg.get("is_final"))
            if text:
                if final:
                    self.finals.append(text)
                if self._cb is not None:
                    self._cb(text, final)

    async def close(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
            if self._reader is not None:
                await asyncio.wait_for(self._reader, timeout=10)
        except Exception:
            pass
        finally:
            await self._ws.close()
            self._ws = None
