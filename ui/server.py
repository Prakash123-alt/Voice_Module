"""Local web UI server.

    python -m ui.server [--host 127.0.0.1] [--port 8765] [--no-browser]

FastAPI app that runs the VoiceAssistant in-process, streams its events to the browser over a
WebSocket, and exposes REST endpoints for control, settings, enrollment and calibration.
Audio stays in Python (sounddevice); the browser never receives raw audio or the API key.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import secrets
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from voice.config import Config, load_config
from voice.dsp import rms_db
from voice.logging_utils import log, setup_logging
from voice.state import State

from .events import BusLogHandler, EventBus

logger = logging.getLogger("ui")
STATIC = Path(__file__).parent / "static"
# Settings the browser may change. API key / attribution come only from .env.
READONLY = {("openrouter", "api_key"), ("openrouter", "site_url"), ("openrouter", "app_name")}


# ----------------------------------------------------------------------------- config helpers
def config_to_dict(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    d["openrouter"].pop("api_key", None)
    return d


def _coerce(value: Any, hint: Any, name: str) -> Any:
    """Validate/convert a JSON value against a dataclass field type hint."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Union:
        if value is None and type(None) in args:
            return None
        for a in (a for a in args if a is not type(None)):
            try:
                return _coerce(value, a, name)
            except ValueError:
                continue
        raise ValueError(f"{name}: invalid value {value!r}")
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name}: expected true/false")
    if hint is int:
        if isinstance(value, bool):
            raise ValueError(f"{name}: expected integer")
        if isinstance(value, (int, float)) and float(value).is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ValueError(f"{name}: expected integer")
    if hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ValueError(f"{name}: expected number")
    if hint is str:
        if isinstance(value, str):
            return value
        raise ValueError(f"{name}: expected text")
    if hint is dict or origin is dict:
        if isinstance(value, dict):
            return value
        raise ValueError(f"{name}: expected object")
    return value


def validate_patch(patch: dict) -> dict:
    """Validate {section: {key: value}} against the Config dataclasses; returns coerced patch."""
    sections = get_type_hints(Config)
    out: dict[str, dict] = {}
    for section, values in patch.items():
        if section not in sections or not isinstance(values, dict):
            raise ValueError(f"unknown section {section!r}")
        hints = get_type_hints(sections[section])
        for key, val in values.items():
            if key not in hints:
                raise ValueError(f"unknown setting {section}.{key}")
            if (section, key) in READONLY:
                raise ValueError(f"{section}.{key} can only be set in .env")
            out.setdefault(section, {})[key] = _coerce(val, hints[key], f"{section}.{key}")
    _check_ranges(out)
    return out


def _check_ranges(p: dict) -> None:
    def rng(sec, key, lo, hi):
        v = p.get(sec, {}).get(key)
        if v is not None and not (lo <= v <= hi):
            raise ValueError(f"{sec}.{key} must be between {lo} and {hi}")
    rng("speaker", "threshold", 0.0, 1.0)
    rng("barge_in", "threshold", 0.0, 1.0)
    rng("vad", "threshold", 0.0, 1.0)
    rng("llm", "temperature", 0.0, 2.0)
    rng("audio", "input_gain", 0.0, 10.0)
    rng("audio", "output_gain", 0.0, 10.0)
    for sec, key in (("vad", "min_speech_ms"), ("vad", "hangover_ms"), ("vad", "pad_ms"),
                     ("barge_in", "min_speech_ms"), ("speaker", "min_segment_ms")):
        rng(sec, key, 0, 10000)
    for sec, key, allowed in (("aec", "backend", {"webrtc", "speex", "gate", "none"}),
                              ("stt", "mode", {"local", "openrouter"}), ("tts", "mode", {"local", "openrouter"}),
                              ("denoise", "backend", {"rnnoise", "deepfilternet", "none"}),
                              ("speaker", "backend", {"resemblyzer", "ecapa"})):
        v = p.get(sec, {}).get(key)
        if v is not None and v not in allowed:
            raise ValueError(f"{sec}.{key} must be one of {sorted(allowed)}")


def write_config_patch(path: Path, patch: dict) -> None:
    """Apply a validated patch to config.yaml, preserving comments and layout (ruamel round-trip)."""
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # don't re-wrap long scalars (system_prompt) when round-tripping
    data = yaml.load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    for section, values in patch.items():
        if section not in data or data[section] is None:
            data[section] = {}
        for k, v in values.items():
            data[section][k] = v
    tmp = path.with_suffix(".yaml.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    tmp.replace(path)


# ----------------------------------------------------------------------------- controller
class Controller:
    """Owns the assistant lifecycle and the enrollment session."""

    def __init__(self, bus: EventBus, cfg_path: Path, env_file: Path,
                 factory: Optional[Callable[[Config], Any]] = None) -> None:
        self.bus, self.cfg_path, self.env_file = bus, cfg_path, env_file
        self.factory = factory
        self.assistant = None
        self.task: Optional[asyncio.Task] = None
        self.status = "stopped"          # stopped | starting | running | stopping | error
        self.error: Optional[str] = None
        self._lock = asyncio.Lock()
        self._embedder = None
        self.enroll_embs: list[np.ndarray] = []
        self.busy_audio = False          # enrollment / mic test is using the mic

    def cfg(self) -> Config:
        return load_config(self.cfg_path, self.env_file)

    def _set_status(self, status: str, error: Optional[str] = None) -> None:
        self.status, self.error = status, error
        self.bus.publish("status", **self.snapshot())

    def snapshot(self) -> dict:
        cfg = self.cfg()
        a = self.assistant
        profile = Path(cfg.speaker.profile_path)
        return {
            "status": self.status, "error": self.error,
            "state": a.sm.state.value if a else State.IDLE.value,
            "muted": bool(a and a.muted),
            "aec": type(a.aec).__name__ if a else None,
            "barge_in": bool(a and cfg.barge_in.enabled and a.aec.available),
            "profile": profile.exists(),
            "api_key_set": bool(cfg.openrouter.api_key),
            "models": {"llm": cfg.llm.model, "stt": f"{cfg.stt.mode}:{cfg.stt.local_model if cfg.stt.mode == 'local' else cfg.stt.openrouter_model}",
                       "tts": f"{cfg.tts.mode}:{Path(cfg.tts.piper_model).stem if cfg.tts.mode == 'local' else cfg.tts.openrouter_model}"},
            "history": list(a.history) if a else [],
        }

    def _default_factory(self, cfg: Config):
        from main import build_assistant
        return build_assistant(cfg)

    async def start(self) -> None:
        async with self._lock:
            if self.status in ("running", "starting"):
                return
            if self.busy_audio:
                raise HTTPException(409, "Microphone is busy (enrollment / mic test in progress)")
            # fail fast, before spending ~15 s loading models (only for the real pipeline)
            if self.factory is None and not Path(self.cfg().speaker.profile_path).exists():
                raise HTTPException(409, "No voice profile yet - enroll your voice first (Voice profile page).")
            self._set_status("starting")
            try:
                cfg = self.cfg()
                a = await asyncio.to_thread(self.factory or self._default_factory, cfg)
            except Exception as e:
                msg = str(e) if isinstance(e, (FileNotFoundError, RuntimeError, ValueError)) else f"{e.__class__.__name__}: {e}"
                log(logger, logging.WARNING, "assistant failed to start", err=msg)  # the API response reports it
                self._set_status("error", msg)
                return
            a.on_event = lambda kind, data: self.bus.publish_threadsafe(kind, **data)
            self.assistant = a
            self.task = asyncio.create_task(a.run(), name="assistant")
            self.task.add_done_callback(self._on_done)
            self._set_status("running")

    def _on_done(self, task: asyncio.Task) -> None:
        err = None
        if not task.cancelled() and task.exception() is not None:
            err = f"{task.exception().__class__.__name__}: {task.exception()}"
            log(logger, logging.ERROR, "assistant stopped with error", err=err)
        self.assistant, self.task = None, None
        self._set_status("error" if err else "stopped", err)

    async def stop(self) -> None:
        async with self._lock:
            if not self.task:
                return
            self._set_status("stopping")
            self.assistant.stop()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=10)
            except asyncio.TimeoutError:
                self.task.cancel()
            except Exception:
                pass

    async def restart_if_running(self) -> bool:
        if self.status != "running":
            return False
        await self.stop()
        await self.start()
        return True

    def embedder(self):
        if self._embedder is None:
            from voice.speaker import build_embedder
            self._embedder = build_embedder(self.cfg().speaker.backend)
        return self._embedder

    def _live_process(self, audio: np.ndarray) -> np.ndarray:
        from enroll import live_process
        return live_process(audio, self.cfg())

    def record(self, seconds: float, event: str) -> np.ndarray:
        """Blocking mic capture (run in a thread) that streams levels to the UI as `event`."""
        import sounddevice as sd
        from voice.audio_io import _resolve_device
        cfg = self.cfg()
        rate = cfg.audio.sample_rate
        chunks: list[np.ndarray] = []

        def cb(indata, frames, t, status):
            x = indata[:, 0].copy()
            chunks.append(x)
            self.bus.publish_threadsafe(event, db=round(rms_db(x), 1), peak=round(float(np.max(np.abs(x))), 3))

        from enroll import STARTUP_DISCARD_S
        skip = int(STARTUP_DISCARD_S * rate)
        with sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=rate // 20,
                            device=_resolve_device(cfg.audio.input_device, "input"), callback=cb):
            time.sleep(seconds + STARTUP_DISCARD_S)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return audio[skip: skip + int(seconds * rate)] * cfg.audio.input_gain  # drop the stream-open click


# ----------------------------------------------------------------------------- app
def create_app(cfg_path: str | Path = "config.yaml", env_file: str | Path = ".env",
               factory: Optional[Callable[[Config], Any]] = None, token: Optional[str] = None,
               attach_logging: bool = True) -> FastAPI:
    bus = EventBus()
    ctl = Controller(bus, Path(cfg_path), Path(env_file), factory)
    handler = BusLogHandler(bus) if attach_logging else None

    async def metrics_loop() -> None:
        while True:
            await asyncio.sleep(1.0)
            a = ctl.assistant
            if a is not None:
                bus.publish("metrics", summary=a.metrics.summary())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.bind(asyncio.get_running_loop())
        if handler:
            logging.getLogger().addHandler(handler)
        mt = asyncio.create_task(metrics_loop())
        try:
            yield
        finally:
            mt.cancel()
            await ctl.stop()
            if handler:
                logging.getLogger().removeHandler(handler)

    app = FastAPI(title="VoiceModule UI", lifespan=lifespan)
    app.state.ctl, app.state.bus = ctl, bus

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token and request.url.path.startswith("/api/"):
            supplied = request.headers.get("x-ui-token") or request.query_params.get("token")
            if not supplied or not secrets.compare_digest(supplied, token):
                return JSONResponse({"detail": "invalid or missing UI token"}, status_code=401)
        return await call_next(request)

    def need_assistant():
        if ctl.assistant is None or ctl.status != "running":
            raise HTTPException(409, "Assistant is not running - press Start first")
        return ctl.assistant

    def need_idle_mic():
        if ctl.status in ("running", "starting"):
            raise HTTPException(409, "Stop the assistant first (it is using the microphone)")
        if ctl.busy_audio:
            raise HTTPException(409, "Microphone busy")

    # --- status / control
    @app.get("/api/status")
    async def status():
        return ctl.snapshot()

    @app.post("/api/start")
    async def start():
        await ctl.start()
        if ctl.status == "error":
            raise HTTPException(500, ctl.error or "failed to start")
        return ctl.snapshot()

    @app.post("/api/stop")
    async def stop():
        await ctl.stop()
        return ctl.snapshot()

    @app.post("/api/text")
    async def text(body: dict = Body(...)):
        a = need_assistant()
        msg = str(body.get("text", ""))[:2000]
        if not a.submit_text(msg):
            raise HTTPException(409, f"Busy ({a.sm.state.value}) - wait until it is listening")
        return {"ok": True}

    @app.post("/api/mute")
    async def mute(body: dict = Body(...)):
        a = need_assistant()
        a.muted = bool(body.get("muted"))
        bus.publish("status", **ctl.snapshot())
        return {"muted": a.muted}

    @app.post("/api/history/clear")
    async def clear_history():
        if ctl.assistant:
            ctl.assistant.history.clear()
        bus.clear_replay({"transcript", "turn", "interrupted_reply", "speaker", "bargein"})
        return {"ok": True}

    # --- config
    @app.get("/api/config")
    async def get_config():
        cfg = ctl.cfg()
        return {"config": config_to_dict(cfg), "api_key_set": bool(cfg.openrouter.api_key)}

    @app.put("/api/config")
    async def put_config(patch: dict = Body(...)):
        try:
            clean = validate_patch(patch)
        except ValueError as e:
            raise HTTPException(422, str(e))
        await asyncio.to_thread(write_config_patch, ctl.cfg_path, clean)
        ctl._embedder = None if "speaker" in clean else ctl._embedder
        log(logger, logging.INFO, "config updated", keys=",".join(f"{s}.{k}" for s, v in clean.items() for k in v))
        restarted = await ctl.restart_if_running()
        cfg = ctl.cfg()
        return {"config": config_to_dict(cfg), "api_key_set": bool(cfg.openrouter.api_key), "restarted": restarted}

    @app.get("/api/devices")
    async def devices():
        import sounddevice as sd
        hostapis = [h["name"] for h in sd.query_hostapis()]
        inputs, outputs = [], []
        for i, d in enumerate(sd.query_devices()):
            item = {"index": i, "name": d["name"].split("\n")[0], "hostapi": hostapis[d["hostapi"]]}
            if d["max_input_channels"] > 0:
                inputs.append(item)
            if d["max_output_channels"] > 0:
                outputs.append(item)
        default_in, default_out = sd.default.device
        return {"inputs": inputs, "outputs": outputs, "default_input": default_in, "default_output": default_out}

    @app.post("/api/check")
    async def check():
        from voice.openrouter import OpenRouterClient
        cfg = ctl.cfg()
        if not cfg.openrouter.api_key:
            raise HTTPException(400, "OPENROUTER_API_KEY is not set in .env")
        client = OpenRouterClient(cfg.openrouter)
        try:
            models = set(await client.list_models())
        except Exception as e:
            raise HTTPException(502, f"Could not reach OpenRouter: {e}")
        finally:
            await client.aclose()
        wanted = {"llm": cfg.llm.model}
        if cfg.stt.mode == "openrouter":
            wanted["stt"] = cfg.stt.openrouter_model
        if cfg.tts.mode == "openrouter":
            wanted["tts"] = cfg.tts.openrouter_model
        res = {}
        for role, slug in wanted.items():
            vendor = slug.split("/")[0]
            res[role] = {"slug": slug, "ok": slug in models,
                         "candidates": sorted(m for m in models if m.startswith(vendor + "/"))[:8]}
        return {"models": res, "total": len(models)}

    # --- mic test / enrollment / calibration
    @app.post("/api/mictest")
    async def mictest(body: dict = Body(default={})):
        need_idle_mic()
        ctl.busy_audio = True
        try:
            audio = await asyncio.to_thread(ctl.record, float(body.get("seconds", 3)), "mic_level")
        except Exception as e:
            raise HTTPException(500, f"Mic error: {e}")
        finally:
            ctl.busy_audio = False
        from enroll import check_take
        ok, problem, level = check_take(audio)
        return {"ok": ok, "problem": problem, "level_db": round(level, 1),
                "peak": round(float(np.max(np.abs(audio))) if audio.size else 0.0, 3)}

    @app.post("/api/enroll/start")
    async def enroll_start():
        need_idle_mic()
        from enroll import PHRASES
        ctl.enroll_embs = []
        await asyncio.to_thread(ctl.embedder)  # load model up-front so the first take isn't slow
        return {"phrases": PHRASES, "min": 5, "max": 10}

    @app.post("/api/enroll/record")
    async def enroll_record(body: dict = Body(default={})):
        need_idle_mic()
        from enroll import check_take
        seconds = min(8.0, max(2.0, float(body.get("seconds", 4))))
        ctl.busy_audio = True
        try:
            audio = await asyncio.to_thread(ctl.record, seconds, "enroll_level")
        except Exception as e:
            raise HTTPException(500, f"Mic error: {e}")
        finally:
            ctl.busy_audio = False
        ok, problem, level = check_take(audio)
        if ok:
            ctl.enroll_embs.append(await asyncio.to_thread(ctl.embedder().embed, ctl._live_process(audio)))
        return {"ok": ok, "problem": problem, "level_db": round(level, 1), "count": len(ctl.enroll_embs)}

    @app.post("/api/enroll/undo")
    async def enroll_undo():
        if ctl.enroll_embs:
            ctl.enroll_embs.pop()
        return {"count": len(ctl.enroll_embs)}

    @app.post("/api/enroll/finish")
    async def enroll_finish():
        if len(ctl.enroll_embs) < 5:
            raise HTTPException(400, f"Need at least 5 good takes (have {len(ctl.enroll_embs)})")
        from calibrate import leave_one_out
        from enroll import save_profile
        from voice.speaker import suggest_threshold
        cfg = ctl.cfg()
        meta = await asyncio.to_thread(save_profile, ctl.enroll_embs, cfg)
        genuine = leave_one_out(ctl.enroll_embs)
        rep = suggest_threshold(genuine)
        log(logger, logging.INFO, "enrollment saved", phrases=meta["phrases"],
            self_sim=round(meta["self_similarity_mean"], 3))
        bus.publish("status", **ctl.snapshot())
        return {"meta": meta, "calibration": {**rep, "genuine": [round(g, 4) for g in genuine],
                                              "current": cfg.speaker.threshold}}

    @app.get("/api/calibrate")
    async def calibrate():
        from calibrate import leave_one_out
        from voice.speaker import suggest_threshold
        cfg = ctl.cfg()
        p = Path(cfg.speaker.profile_path).with_name("enroll_embeddings.npy")
        if not p.exists():
            raise HTTPException(404, "No enrollment yet")
        genuine = leave_one_out(list(np.load(p)))
        return {**suggest_threshold(genuine), "genuine": [round(g, 4) for g in genuine],
                "current": cfg.speaker.threshold}

    # --- websocket
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        if token and websocket.query_params.get("token") != token:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        q = bus.subscribe()
        await websocket.send_json({"type": "status", "ts": time.time(), **ctl.snapshot()})

        async def drain_incoming():
            while True:
                await websocket.receive_text()  # we don't expect messages; this detects disconnects

        reader = asyncio.create_task(drain_incoming())
        try:
            while True:
                getter = asyncio.create_task(q.get())
                done, _ = await asyncio.wait({getter, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    getter.cancel()
                    break
                await websocket.send_json(getter.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()
            bus.unsubscribe(q)

    if STATIC.exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="VoiceModule web UI")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your LAN")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.json)
    token = os.environ.get("UI_TOKEN") or None
    if args.host not in ("127.0.0.1", "localhost") and not token:
        log(logger, logging.WARNING, "UI exposed on the network without UI_TOKEN - anyone on your LAN can control it")
    app = create_app(args.config, token=token)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}" + (f"/?token={token}" if token else "")
    log(logger, logging.INFO, "web UI", url=url)
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
