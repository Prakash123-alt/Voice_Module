import asyncio
import json
import sys
import types

import numpy as np
import pytest

from speakerlock.audio_io import from_pcm16, to_pcm16
from speakerlock.pipeline import ChunkMeta

SR = 16000


def meta(verified, vad, n=512):
    return ChunkMeta(verified, 0.8 if verified else 0.2, "me", 60.0, 1.0, vad, 0, n, 0)


# ------------------------------------------------------------------ whisper adapter
def test_whisper_adapter_segments_on_pause_and_skips_unverified():
    from speakerlock.integrations.whisper_adapter import WhisperAdapter
    heard = []
    ad = WhisperAdapter(transcriber=lambda a: heard.append(len(a)) or f"utt{len(heard)}", end_silence_ms=320)
    out = []
    speech, quiet = np.full(512, 0.1, np.float32), np.zeros(512, np.float32)
    for _ in range(30):                       # ~1 s verified speech
        out += ad.feed(speech, meta(True, 0.9))
    for _ in range(12):                       # pause > 320 ms -> utterance ends
        out += ad.feed(quiet, meta(True, 0.0))
    for _ in range(30):                       # unverified speaker: never buffered
        out += ad.feed(speech, meta(False, 0.9))
    out += ad.flush()
    assert out == ["utt1"] and heard == [30 * 512]


def test_whisper_adapter_gate_close_ends_utterance():
    from speakerlock.integrations.whisper_adapter import WhisperAdapter
    ad = WhisperAdapter(transcriber=lambda a: "x", end_silence_ms=5000)
    for _ in range(20):
        ad.feed(np.full(512, 0.1, np.float32), meta(True, 0.9))
    assert ad.feed(np.zeros(512, np.float32), meta(False, 0.0)) == ["x"]


def test_whisper_adapter_real_model(clips):
    pytest.importorskip("faster_whisper")
    from speakerlock.integrations.whisper_adapter import WhisperAdapter, faster_whisper_transcriber
    ad = WhisperAdapter(transcriber=faster_whisper_transcriber("tiny.en"))
    x = clips["target_query"]
    out = []
    for i in range(0, len(x), 512):
        out += ad.feed(x[i:i + 512], meta(True, 0.9, len(x[i:i + 512])))
    out += ad.flush()
    assert "chief minister" in " ".join(out).lower()


# ------------------------------------------------------------------ deepgram adapter (local fake server)
def test_deepgram_streamer_protocol():
    pytest.importorskip("websockets")
    from websockets.asyncio.server import serve

    from speakerlock.integrations.deepgram_adapter import DeepgramStreamer
    got = {"bytes": 0, "auth": None, "query": None, "closed": False}

    async def handler(ws):
        got["auth"] = ws.request.headers.get("Authorization")
        got["query"] = ws.request.path
        async for msg in ws:
            if isinstance(msg, bytes):
                got["bytes"] += len(msg)
                if got["bytes"] >= 3200 and not got.get("sent"):
                    got["sent"] = True
                    await ws.send(json.dumps({"type": "Results", "is_final": False,
                                              "channel": {"alternatives": [{"transcript": "hello"}]}}))
                    await ws.send(json.dumps({"type": "Results", "is_final": True,
                                              "channel": {"alternatives": [{"transcript": "hello world"}]}}))
            elif json.loads(msg).get("type") == "CloseStream":
                got["closed"] = True
                await ws.send(json.dumps({"type": "Metadata"}))
                await ws.close()

    async def main():
        seen = []
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with DeepgramStreamer("test-key", on_transcript=lambda t, f: seen.append((t, f)),
                                        url=f"ws://127.0.0.1:{port}/v1/listen") as dg:
                for _ in range(5):
                    await dg.send(np.full(512, 0.1, np.float32))
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.2)
            return seen, dg.finals

    seen, finals = asyncio.run(main())
    assert got["auth"] == "Token test-key" and got["bytes"] == 5 * 512 * 2 and got["closed"]
    assert "encoding=linear16" in got["query"] and "sample_rate=16000" in got["query"]
    assert seen == [("hello", False), ("hello world", True)] and finals == ["hello world"]


def test_deepgram_requires_key():
    from speakerlock.integrations.deepgram_adapter import DeepgramStreamer
    with pytest.raises(ValueError):
        DeepgramStreamer("")


# ------------------------------------------------------------------ pipecat filter
def _pipecat_stub(monkeypatch):
    """Minimal stand-in for pipecat's filter API when pipecat isn't installed."""
    base = types.ModuleType("pipecat.audio.filters.base_audio_filter")
    base.BaseAudioFilter = type("BaseAudioFilter", (), {})
    frames = types.ModuleType("pipecat.frames.frames")
    frames.FilterControlFrame = type("FilterControlFrame", (), {})
    frames.FilterEnableFrame = type("FilterEnableFrame", (frames.FilterControlFrame,),
                                    {"__init__": lambda self, enable: setattr(self, "enable", enable)})
    for name in ("pipecat", "pipecat.audio", "pipecat.audio.filters", "pipecat.frames"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, frames.__name__, frames)
    monkeypatch.delitem(sys.modules, "speakerlock.integrations.pipecat_filter", raising=False)


def test_pipecat_filter_roundtrip(monkeypatch):
    try:
        import pipecat.audio.filters.base_audio_filter  # noqa: F401
    except ImportError:
        _pipecat_stub(monkeypatch)
    from pipecat.frames.frames import FilterEnableFrame

    from speakerlock.integrations.pipecat_filter import SpeakerLockFilter

    class Lock:
        def __init__(self):
            self.n = 0

        def process(self, x):
            self.n += len(x)
            return x * 0.5, meta(True, 0.9, len(x))

        def close(self):
            pass

    metas = []
    f = SpeakerLockFilter(user_id="me", lock=Lock(), on_meta=metas.append)

    async def run():
        await f.start(16000)
        x = np.full(320, 0.2, np.float32)
        y1 = from_pcm16(await f.filter(to_pcm16(x)))
        await f.process_frame(FilterEnableFrame(False))
        y2 = from_pcm16(await f.filter(to_pcm16(x)))
        await f.stop()
        return y1, y2

    y1, y2 = asyncio.run(run())
    assert np.allclose(y1, 0.1, atol=1e-3) and np.allclose(y2, 0.2, atol=1e-3) and len(metas) == 1


# ------------------------------------------------------------------ livekit adapter (duck-typed frame)
def test_livekit_adapter_roundtrip():
    from speakerlock.integrations.livekit_adapter import LiveKitSpeakerLock

    class Frame:
        def __init__(self, data=b"", sample_rate=16000, num_channels=1, samples_per_channel=0):
            self.data, self.sample_rate = data, sample_rate
            self.num_channels, self.samples_per_channel = num_channels, samples_per_channel

    class Lock:
        def process(self, x):
            return x * 0.5, meta(True, 0.9, len(x))

        def close(self):
            pass

    metas = []
    ad = LiveKitSpeakerLock(user_id="me", lock=Lock(), on_meta=metas.append)
    f = Frame(data=to_pcm16(np.full(320, 0.4, np.float32)), samples_per_channel=320)
    out = ad.process(f)
    assert isinstance(out, Frame) and out.sample_rate == 16000 and out.samples_per_channel == 320
    assert np.allclose(from_pcm16(out.data), 0.2, atol=1e-3)
    assert ad.last_meta is metas[0] and metas[0].verified
    ad.close()


def test_livekit_adapter_rejects_multichannel():
    from speakerlock.integrations.livekit_adapter import LiveKitSpeakerLock

    class Frame:
        def __init__(self, data=b"", sample_rate=16000, num_channels=2, samples_per_channel=0):
            self.data, self.sample_rate, self.num_channels = data, sample_rate, num_channels
            self.samples_per_channel = samples_per_channel

    class Lock:
        def process(self, x):
            return x, meta(True, 0.9, len(x))

    ad = LiveKitSpeakerLock(user_id="me", lock=Lock())
    with pytest.raises(ValueError):
        ad.process(Frame(data=to_pcm16(np.zeros(320, np.float32))))


# ------------------------------------------------------------------ liveness
def test_liveness_score_bounds_and_inputs():
    from speakerlock.liveness import liveness_score
    assert liveness_score(np.zeros(16000, np.float32)) == 0.5          # silence: neutral
    t = np.arange(16000) / 16000
    voiced = (0.15 * np.sin(2 * np.pi * 150 * t) * (1 + 0.4 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)
    live = liveness_score(voiced)
    assert 0.0 <= live <= 1.0
    muffled = liveness_score(np.convolve(voiced, np.ones(40) / 40, "same").astype(np.float32))
    assert live >= muffled                                            # HF roll-off lowers the estimate
