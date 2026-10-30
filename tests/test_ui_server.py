"""Web UI server: config safety/validation, lifecycle, and the WebSocket event stream (fake pipeline)."""
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice.assistant import Event, VoiceAssistant
from voice.config import Config
from ui.server import _coerce, create_app, validate_patch

from tests.test_assistant_bargein import (FakeOut, FakeTTS, FakeSTT, LevelVAD, LevelVerifier, NullIn, PassAEC,
                                          PassDenoise, ShortLLM, USER)

SECRET = "sk-or-v1-THIS-MUST-NEVER-REACH-THE-BROWSER-0123456789"
ROOT = Path(__file__).resolve().parents[1]


class SilentIn(NullIn):
    """A mic that delivers silence in real time (NullIn returns None, which looks like an unplugged device)."""
    def read(self, timeout=None):
        time.sleep(0.02)
        return np.zeros(320, dtype=np.float32)


def fake_factory(cfg: Config) -> VoiceAssistant:
    return VoiceAssistant(cfg, audio_in=SilentIn(), audio_out=FakeOut(), aec=PassAEC(), denoiser=PassDenoise(),
                          vad=LevelVAD(), verifier=LevelVerifier(), stt=FakeSTT(), llm=ShortLLM(), tts=FakeTTS())


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config.yaml", cfg)
    env = tmp_path / ".env"
    env.write_text(f"OPENROUTER_API_KEY={SECRET}\n")
    yield cfg, env
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def client(paths):
    cfg, env = paths
    app = create_app(cfg, env, factory=fake_factory, attach_logging=False)
    with TestClient(app) as c:
        yield c


def test_config_never_contains_api_key(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    assert SECRET not in r.text and "api_key\"" not in r.text.replace("api_key_set", "")
    assert r.json()["api_key_set"] is True
    s = client.get("/api/status")
    assert SECRET not in s.text and s.json()["api_key_set"] is True


def test_put_config_validates_and_preserves_comments(client, paths):
    cfg_path, _ = paths
    assert client.put("/api/config", json={"speaker": {"threshold": "high"}}).status_code == 422
    assert client.put("/api/config", json={"speaker": {"threshold": 1.5}}).status_code == 422
    assert client.put("/api/config", json={"openrouter": {"api_key": "x"}}).status_code == 422
    assert client.put("/api/config", json={"nope": {"a": 1}}).status_code == 422
    assert client.put("/api/config", json={"aec": {"backend": "magic"}}).status_code == 422

    r = client.put("/api/config", json={"speaker": {"threshold": 0.71}, "vad": {"hangover_ms": 400},
                                        "audio": {"input_device": None}, "aec": {"delay_ms": 120}})
    assert r.status_code == 200, r.text
    body = r.json()["config"]
    assert body["speaker"]["threshold"] == 0.71 and body["vad"]["hangover_ms"] == 400
    assert body["audio"]["input_device"] is None and body["aec"]["delay_ms"] == 120
    text = cfg_path.read_text(encoding="utf-8")
    assert "# cosine similarity; see calibrate.py" in text  # comments survive the round-trip
    assert SECRET not in text


def test_coerce_union_and_types():
    from typing import Optional, Union
    assert _coerce("auto", Union[int, str], "x") == "auto"
    assert _coerce(120, Union[int, str], "x") == 120
    assert _coerce(None, Optional[Union[int, str]], "x") is None
    assert _coerce(3, float, "x") == 3.0
    with pytest.raises(ValueError):
        _coerce(True, int, "x")
    with pytest.raises(ValueError):
        validate_patch({"llm": {"temperature": 5}})


def test_text_requires_running_assistant(client):
    r = client.post("/api/text", json={"text": "hi"})
    assert r.status_code == 409


def recv_until(ws, pred, timeout=5.0):
    seen = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = ws.receive_json()
        seen.append(ev)
        if pred(ev):
            return ev, seen
    raise AssertionError(f"event not received; got {[e['type'] for e in seen]}")


def test_lifecycle_and_event_stream(client):
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "status" and hello["status"] == "stopped"
        assert SECRET not in str(hello)

        r = client.post("/api/start")
        assert r.status_code == 200 and r.json()["status"] == "running"
        recv_until(ws, lambda e: e["type"] == "state" and e["to"] == "LISTENING")

        assert client.post("/api/text", json={"text": "what is the capital of France"}).status_code == 200
        recv_until(ws, lambda e: e["type"] == "transcript" and e["role"] == "user" and e["source"] == "text")
        ev, seen = recv_until(ws, lambda e: e["type"] == "transcript" and e["role"] == "assistant")
        assert "Paris" in ev["text"]
        types = [e["type"] for e in seen]
        assert "reply_delta" in types and "turn" in types
        # first audio flips to SPEAKING before the full reply is logged; FakeOut never drains, so it stays there
        assert any(e["type"] == "state" and e["to"] == "SPEAKING" for e in seen)
        # no manual interrupt exists: a verified-voice barge candidate is the only way in
        assert client.post("/api/interrupt").status_code in (404, 405)
        a = client.app.state.ctl.assistant
        window = np.full(16000, USER, dtype=np.float32)  # 1 s of the enrolled voice
        a._loop.call_soon_threadsafe(a._events.put_nowait,
                                     Event("barge_candidate", audio=window, info={"speech_ms": 600}))
        ev, _ = recv_until(ws, lambda e: e["type"] == "bargein")
        assert "manual" not in ev

        assert client.post("/api/mute", json={"muted": True}).json()["muted"] is True
        r = client.post("/api/stop")
        assert r.status_code == 200
        recv_until(ws, lambda e: e["type"] == "status" and e["status"] == "stopped")


def test_start_failure_reports_error(paths):
    cfg, env = paths

    def broken(_cfg):
        raise FileNotFoundError("speaker profile profile/voice.npy missing - run `python enroll.py` first")

    with TestClient(create_app(cfg, env, factory=broken, attach_logging=False)) as c:
        r = c.post("/api/start")
        assert r.status_code == 500 and "enroll.py" in r.json()["detail"]
        assert c.get("/api/status").json()["status"] == "error"


def test_real_pipeline_start_without_profile_fails_fast(paths):
    cfg, env = paths
    text = cfg.read_text(encoding="utf-8").replace("profile/voice.npy", str(cfg.parent / "missing.npy").replace("\\", "/"))
    cfg.write_text(text, encoding="utf-8")
    with TestClient(create_app(cfg, env, attach_logging=False)) as c:  # default factory = real models
        t0 = time.time()
        r = c.post("/api/start")
        assert r.status_code == 409 and "enroll" in r.json()["detail"].lower()
        assert time.time() - t0 < 2                      # no model loading happened
        assert c.get("/api/status").json()["status"] == "stopped"


def test_token_auth(paths):
    cfg, env = paths
    with TestClient(create_app(cfg, env, factory=fake_factory, token="s3cret", attach_logging=False)) as c:
        assert c.get("/api/status").status_code == 401
        assert c.get("/api/status", headers={"X-UI-Token": "s3cret"}).status_code == 200
        assert c.get("/api/status?token=s3cret").status_code == 200


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VoiceModule" in r.text
    assert client.get("/app.js").status_code == 200
