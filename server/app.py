"""Gemini-Live-style duplex server: `python -m server.app` -> http://127.0.0.1:8766

One WebSocket `/ws` per session. Binary frames = int16 mono PCM both ways
(client->server 16 kHz mic, server->client 16 kHz TTS). JSON text frames carry
events: state, transcript, speaker, duck/restore, interrupted, speech_started,
turn_complete, error. Serves the web client from web/.
"""
from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voice.config import load_config
from voice.denoise import build_denoiser
from voice.openrouter import OpenRouterClient
from voice.speaker import ProfileSpeakerVerifier, build_embedder
from voice.stt import build_stt
from voice.tts import build_tts
from voice.vad import Segmenter, build_vad

from .interrupt import InterruptConfig
from .session import Session

logger = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent.parent / "web"


def build(cfg):
    """Load the local engines once; shared by all sessions (they're stateless apart
    from the speaker profile)."""
    client = OpenRouterClient(cfg.openrouter)
    stt = build_stt(cfg.stt, client, cfg.audio.sample_rate)
    tts = build_tts(cfg.tts, client, cfg.audio.sample_rate)
    vad = build_vad(cfg.audio.sample_rate)
    try:
        verifier = ProfileSpeakerVerifier.from_file(
            build_embedder(cfg.speaker.backend), cfg.speaker.profile_path,
            threshold=cfg.speaker.threshold, min_segment_ms=cfg.speaker.min_segment_ms)
    except FileNotFoundError:
        logger.warning("no speaker profile - running open (enroll via main app)")
        verifier = None
    return client, stt, tts, vad, verifier


def make_segmenter(cfg) -> Segmenter:
    return Segmenter(frame_ms=cfg.audio.frame_ms, threshold=cfg.vad.threshold,
                     min_speech_ms=cfg.vad.min_speech_ms, hangover_ms=cfg.vad.hangover_ms,
                     pad_ms=cfg.vad.pad_ms, max_segment_s=cfg.vad.max_segment_s)


def create_app(cfg_path: str = "config.yaml", env_file: str = ".env") -> FastAPI:
    cfg = load_config(cfg_path, env_file)
    engines = build(cfg)
    icfg = InterruptConfig(vad_threshold=cfg.vad.threshold,
                           min_score=cfg.barge_in.threshold or cfg.speaker.threshold - 0.25)
    app = FastAPI(title="VoiceModule Live")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        client, stt, tts, vad, verifier = engines
        out_q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)   # ~1 s of audio max
        session = Session(cfg, stt, llm=client, tts=tts, vad=vad, verifier=verifier,
                          segmenter=make_segmenter(cfg),
                          send_json=lambda m: _safe_send(sock, m),
                          send_bytes=lambda b: _enqueue(out_q, b), icfg=icfg)
        pump = asyncio.create_task(session.run())

        async def send_audio():
            while not session.closed:
                try:
                    chunk = await asyncio.to_thread(out_q.get, True, 0.5)
                    await sock.send_bytes(chunk)
                except queue.Empty:
                    continue
                except Exception:
                    return

        sender = asyncio.create_task(send_audio())
        try:
            while True:
                msg = await sock.receive()
                if msg.get("bytes") is not None:
                    session.feed_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    import json
                    session.feed_control(json.loads(msg["text"]))
                elif msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            session.close()
            pump.cancel()
            sender.cancel()

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def _safe_send(sock: WebSocket, m: dict) -> None:
    """Session events may come from worker threads; hop to the loop safely."""
    try:
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.ensure_future(sock.send_json(m)))
    except RuntimeError:
        pass


def _enqueue(q: "queue.Queue[bytes]", b: bytes) -> None:
    try:
        q.put_nowait(b)
    except queue.Full:
        try:
            q.get_nowait()          # drop oldest: never let playback lag > ~1 s
        except queue.Empty:
            pass
        try:
            q.put_nowait(b)
        except queue.Full:
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    app = create_app()
    logger.info("live server url=http://127.0.0.1:8766")
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")


if __name__ == "__main__":
    main()
