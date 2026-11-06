"""Local service wrapper for the Voice Firewall (Phase 4 + React UI).

Binds to 127.0.0.1 only.  Every request passes the entry guard:
  - No foreign browser Origin header (the UI's own origin is allow-listed)
  - Valid bearer token (master token or session-scoped token for /ws)

Endpoints
---------
GET  /                 → React UI (web/dist/index.html) — public GET, Origin-checked
GET  /assets/*         → UI assets — public GET
GET  /worklets/*       → audio worklets — public GET
GET  /health           → {"status": "ok"}  (token)
GET  /metrics          → per-stage latency summary  (token)
POST /authorize        → action authorization decision  (token)
POST /api/session      → {sessionId, token, disclosure}  (token → issues WS token)
WS   /stream           → raw float32-LE PCM streaming  (token)
WS   /ws               → s16le PCM + JSON control protocol for the React UI

/ws protocol (mirrors the Voice project's server/src/protocol.ts)
------------------------------------------------------------------
Client → server
  binary : s16le mono 16 kHz PCM frames
  text   : {"type":"start"} {"type":"stop"} {"type":"text","text":...}
           {"type":"barge_in"} {"type":"mute","muted":bool} {"type":"ping","t":n}

Server → client
  {"type":"ready","sessionId","disclosure"}
  {"type":"state","state":"listening"|"idle"}
  {"type":"sentence","text":...}          — firewall verdict line
  {"type":"speaker_rejected","similarity":x}
  {"type":"error","code","message"}
  {"type":"interrupt"} {"type":"pong","t":n}
"""
from __future__ import annotations

import json
import logging
import struct
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from service.entry_guard import EntryGuardMiddleware, SessionToken
from voice_firewall._config import FirewallConfig
from voice_firewall._logging import log, setup_logging
from voice_firewall._pipeline import Firewall
from voice_firewall._types import PipelineContext, RawSegment
from voice_firewall.vad.segmenter import EnergyVAD, Segmenter
from voice_firewall.words.injection import check_injection

logger = logging.getLogger(__name__)

_VERSION = "0.4.0"
_DISCLOSURE = (
    "This microphone is gated by the Voice Firewall. Only the enrolled "
    "owner's voice is forwarded to the agent; risky actions need a click."
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class AuthorizeRequest(BaseModel):
    action_name: str
    params: dict = {}
    segment_id: Optional[str] = None


class SessionRequest(BaseModel):
    sessionId: Optional[str] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: FirewallConfig, token: SessionToken) -> FastAPI:
    """Build the FastAPI app with entry guard, firewall instance, and all routes."""
    app = FastAPI(title="Voice Firewall", docs_url=None, redoc_url=None)

    fw = Firewall(cfg)
    fw.start()

    app.add_middleware(_ASGIMiddlewareShim, token=token, cfg=cfg)

    dist_dir = Path(__file__).parent.parent / "web" / "dist"

    # ---- API routes (registered BEFORE the SPA fallback so they always win) ----

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": _VERSION})

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse(fw.metrics_summary())

    @app.post("/authorize")
    async def authorize(req: AuthorizeRequest) -> JSONResponse:
        decision = fw.authorize(req.action_name, req.params)
        return JSONResponse({
            "verdict": decision.verdict.value,
            "confirmation_text": decision.confirmation_text,
            "reason": decision.reason,
            "audit_id": decision.audit_id,
            "source_tags": decision.source_tags,
        })

    @app.post("/api/session")
    async def session(req: SessionRequest) -> JSONResponse:
        """Mint a WS-scoped session token.  The master token is never sent
        over a WebSocket URL — the client exchanges it here for a token that
        only authorises /ws and /stream."""
        session_id = req.sessionId or str(uuid.uuid4())
        ws_token = token.issue_session()
        return JSONResponse({
            "sessionId": session_id,
            "token": ws_token,
            "disclosure": _DISCLOSURE,
        })

    # ---- WebSocket: raw float32 streaming (original API) ----

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        log(logger, logging.INFO, "websocket client connected", path="/stream")
        try:
            while True:
                raw = await ws.receive_bytes()
                transcript, audio = _parse_frame(raw)
                tagged = fw.process(audio, transcript=transcript)
                if tagged is None:
                    await ws.send_text(json.dumps({
                        "type": "dropped",
                        "reason": "dropped by guard pipeline",
                    }))
                else:
                    await ws.send_text(json.dumps({
                        "type": "passed",
                        "segment_id": tagged.segment_id,
                        "speaker_score": round(tagged.speaker_score, 4),
                        "source": tagged.source,
                        "transcript_flags": tagged.transcript_flags,
                        "interrupt": tagged.interrupt,
                        "liveness_score": tagged.liveness_score,
                    }))
        except WebSocketDisconnect:
            log(logger, logging.INFO, "websocket client disconnected", path="/stream")
        except Exception as exc:  # noqa: BLE001
            log(logger, logging.ERROR, "websocket error", err=str(exc))
            try:
                await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
            except Exception:
                pass

    # ---- WebSocket: React app protocol (/ws) ----

    @app.websocket("/ws")
    async def ws_agent(ws: WebSocket) -> None:
        """Bridge the Voice project's client protocol onto fw.process().

        s16le PCM in → EnergyVAD → Segmenter → fw.process → mapped events.
        """
        await ws.accept()
        session_id = str(uuid.uuid4())
        vad = EnergyVAD()
        segmenter = Segmenter(frame_ms=cfg.audio.frame_ms)
        pcm_tail = np.zeros(0, dtype=np.float32)
        frame_len = cfg.audio.frame_samples
        running = False
        session_token: str | None = None

        log(logger, logging.INFO, "websocket client connected", path="/ws")

        async def _send(obj: dict) -> None:
            try:
                await ws.send_text(json.dumps(obj))
            except Exception:
                pass

        async def _on_segment(audio: np.ndarray, speech_ms: int) -> None:
            """Run the firewall on a completed utterance and emit verdict."""
            seg = RawSegment(audio=audio, speech_ms=speech_ms)
            tagged = fw.process(audio)  # returns TaggedSegment or None
            if tagged is None:
                await _send({"type": "speaker_rejected", "similarity": 0.0})
                return
            flags = ",".join(tagged.transcript_flags) or "none"
            await _send({
                "type": "sentence",
                "text": (f"🛡 segment passed — speaker={tagged.speaker_score:.2f} "
                         f"src={tagged.source} flags={flags}"),
            })

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if msg.get("bytes") is not None:
                    if not running:
                        continue
                    chunk = np.frombuffer(msg["bytes"], dtype="<i2")
                    pcm = np.concatenate([pcm_tail,
                                          chunk.astype(np.float32) / 32768.0])
                    n_frames = len(pcm) // frame_len
                    for i in range(n_frames):
                        frame = pcm[i * frame_len:(i + 1) * frame_len]
                        prob = vad.probability(frame)
                        ev = segmenter.push(frame, prob)
                        if ev and ev.kind == "end" and ev.audio is not None:
                            await _on_segment(ev.audio, ev.speech_ms)
                    pcm_tail = pcm[n_frames * frame_len:]

                elif msg.get("text") is not None:
                    try:
                        ctrl = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue
                    mtype = ctrl.get("type")

                    if mtype == "start":
                        running = True
                        vad.reset(); segmenter.reset()
                        await _send({"type": "ready", "sessionId": session_id,
                                     "disclosure": _DISCLOSURE})
                        await _send({"type": "state", "state": "listening"})
                    elif mtype == "stop":
                        running = False
                        # flush an in-flight segment tail
                        if segmenter.active:
                            ev = segmenter.push(
                                np.zeros(frame_len * 30, dtype=np.float32), 0.0)
                            if ev and ev.kind == "end" and ev.audio is not None:
                                await _on_segment(ev.audio, ev.speech_ms)
                        await _send({"type": "state", "state": "idle"})
                    elif mtype == "text":
                        flags = check_injection(ctrl.get("text", ""))
                        if flags:
                            await _send({"type": "sentence",
                                         "text": f"🛡 transcript flags: {flags}"})
                        else:
                            await _send({"type": "sentence",
                                         "text": "🛡 transcript clean"})
                    elif mtype == "barge_in":
                        fw.set_assistant_speaking(False)
                        await _send({"type": "interrupt"})
                    elif mtype == "mute":
                        if ctrl.get("muted"):
                            vad.reset(); segmenter.reset()
                    elif mtype == "ping":
                        await _send({"type": "pong", "t": ctrl.get("t", 0)})

        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log(logger, logging.ERROR, "websocket error", err=str(exc), path="/ws")
            await _send({"type": "error", "code": "server", "message": str(exc)})
        finally:
            if session_token:
                token.revoke_session(session_token)
            log(logger, logging.INFO, "websocket client disconnected", path="/ws")

    # ---- Static UI + SPA (registered LAST so API routes win) ----
    if dist_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")
        app.mount("/worklets", StaticFiles(directory=dist_dir / "worklets"), name="worklets")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(dist_dir / "index.html")

        @app.get("/admin")
        @app.get("/admin/{rest:path}")
        async def admin_spa(rest: str = "") -> FileResponse:
            return FileResponse(dist_dir / "index.html")

    return app


# ---------------------------------------------------------------------------
# Frame parsing for /stream
# ---------------------------------------------------------------------------

def _parse_frame(raw: bytes) -> tuple[Optional[str], np.ndarray]:
    """Parse a binary /stream frame into (transcript, float32 audio)."""
    if len(raw) >= 4:
        t_len = struct.unpack(">I", raw[:4])[0]
        if 4 + t_len < len(raw):
            transcript = raw[4:4 + t_len].decode("utf-8", errors="replace")
            audio_bytes = raw[4 + t_len:]
            audio = np.frombuffer(audio_bytes, dtype="<f4").copy()
            return transcript, audio
    audio = np.frombuffer(raw, dtype="<f4").copy()
    return None, audio


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def run(cfg: FirewallConfig | None = None) -> None:
    """Start the service. Binds to 127.0.0.1 only."""
    import uvicorn  # noqa: PLC0415
    if cfg is None:
        cfg = FirewallConfig.from_yaml()
    setup_logging(cfg.logging.level, cfg.logging.json)

    token = SessionToken(cfg.service_token)
    app = create_app(cfg, token)

    log(logger, logging.INFO, "starting service",
        bind=cfg.guards.entry.bind, port=cfg.guards.entry.port,
        loopback_only=True)
    import uvicorn  # noqa: PLC0415
    uvicorn.run(app, host=cfg.guards.entry.bind,
                port=cfg.guards.entry.port, log_level="warning")


# ---------------------------------------------------------------------------
# ASGI middleware shim
# ---------------------------------------------------------------------------

class _ASGIMiddlewareShim:
    def __init__(self, app, token: SessionToken, cfg: FirewallConfig) -> None:
        self._inner = EntryGuardMiddleware(app, token, cfg.guards.entry)

    async def __call__(self, scope, receive, send) -> None:
        await self._inner(scope, receive, send)


if __name__ == "__main__":
    run()
