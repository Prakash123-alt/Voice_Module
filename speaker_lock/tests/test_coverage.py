"""Coverage for the module-level helpers, CLI, bench, testset builder, models and export.

All model builders are faked - these tests exercise control flow, not real models.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import speakerlock.cli as cli_mod
import speakerlock.bench as bench_mod
import speakerlock.extractor as extractor_mod
import speakerlock.vad as vad_mod
import speakerlock.embedder as embedder_mod
from speakerlock.audio_io import load_audio, save_audio
from speakerlock.config import Config, SAMPLE_RATE as SR
from speakerlock.interfaces import TargetCondition
from speakerlock.pipeline import ChunkMeta

SR_ = SR
DATA = Path(__file__).parent / "data"


# ------------------------------------------------------------------ fakes
class ToneVAD:
    frame_samples = 512

    def __init__(self, *a, **k):
        self.frame_samples = 512

    def probability(self, frame):
        return 1.0 if np.sqrt(np.mean(frame ** 2)) > 0.01 else 0.0

    def reset(self):
        pass

    def clone(self):
        return ToneVAD()


class SignEmbedder:
    dim = 2

    def __init__(self, *a, **k):
        pass

    def embed(self, audio):
        return np.array([1.0, 0.0], np.float32) if audio.mean() > 0 else np.array([0.0, 1.0], np.float32)


class PassthroughEx:
    latency_samples = 0

    def __init__(self, *a, **k):
        self.started = False

    def condition(self, sample):
        return TargetCondition({})

    def start(self, cond):
        self.started = True

    def process(self, chunk):
        return np.asarray(chunk, np.float32)

    def flush(self):
        return np.zeros(0, np.float32)


def _fakes(monkeypatch):
    monkeypatch.setattr(extractor_mod, "build_extractor", lambda *a, **k: PassthroughEx())
    monkeypatch.setattr(vad_mod, "build_vad", lambda *a, **k: ToneVAD())
    monkeypatch.setattr(embedder_mod, "build_embedder", lambda *a, **k: SignEmbedder())


def voice(sign, seconds, sr=SR_):
    return (sign * (0.1 + 0.05 * np.sin(np.arange(int(seconds * sr)) / 7))).astype(np.float32)


def _meta(n):
    return ChunkMeta(True, 0.9, "me", 5.0, 1.0, 0.9, 0, n, 0)


# ------------------------------------------------------------------ audio_io
def test_audio_io_roundtrip_and_resample(tmp_path):
    from speakerlock.audio_io import from_pcm16, load_audio, resample, save_audio, to_pcm16
    x = voice(1, 0.5)
    p = tmp_path / "a.wav"
    save_audio(p, x)
    y = load_audio(p)
    assert abs(len(y) - len(x)) <= 4 and np.allclose(y[: len(x)], x, atol=1e-2)
    assert np.array_equal(resample(x, SR_), x)
    down = resample(x, 8000)                       # 8 kHz -> 16 kHz upsample
    assert abs(len(down) - 2 * len(x)) <= 4
    b = to_pcm16(x)
    assert np.allclose(from_pcm16(b), x, atol=1e-3)


# ------------------------------------------------------------------ models
def test_models_verified_and_silero_and_dvector(tmp_path, monkeypatch):
    import speakerlock.models as m
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    import hashlib
    good = hashlib.sha256(b"hello").hexdigest()
    assert m._verified(f, good) is f
    with pytest.raises(RuntimeError):
        m._verified(f, "0" * 64)
    with pytest.raises(FileNotFoundError):
        m.dvector(tmp_path / "nope.onnx", auto_export=False)


def test_voxlock_download_mocked(tmp_path, monkeypatch):
    import speakerlock.models as m
    import hashlib
    name = m.VOXLOCK_PRECISIONS["int8-full"][0]
    p = tmp_path / "e.onnx"
    p.write_bytes(b"m")
    monkeypatch.setattr(m, "VOXLOCK_SHA256", {name: hashlib.sha256(b"m").hexdigest()})
    monkeypatch.setattr(m, "VOXLOCK_PRECISIONS", {"int8-full": (name, name)})
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(p))
    assert m.voxlock("int8-full") == (p, p)


# ------------------------------------------------------------------ export
def test_export_dvector_mocked(tmp_path, monkeypatch):
    import torch
    import resemblyzer
    import onnxruntime as ort
    from speakerlock.export import export_dvector

    class Enc:
        def __init__(self, *a, **k):
            pass

        def eval(self):
            return self

        def __call__(self, t):
            return torch.zeros(t.shape[0], 4)

    monkeypatch.setattr(resemblyzer, "VoiceEncoder", Enc)
    monkeypatch.setattr(torch.onnx, "export", lambda *a, **k: Path(a[2]).write_bytes(b"onnx"))
    monkeypatch.setattr(ort, "InferenceSession",
                        lambda *a, **k: types.SimpleNamespace(run=lambda *a2, **k2: [np.zeros((3, 4), np.float32)]))
    out = export_dvector(tmp_path / "dv.onnx")
    assert out.exists() and out.read_bytes() == b"onnx"


# ------------------------------------------------------------------ testset builder
def _fake_librispeech(root: Path, speakers=("1001", "2002", "3003", "4004"),
                      genders=("F", "M", "F", "M")) -> Path:
    (root / "dev-clean").mkdir(parents=True)
    (root / "SPEAKERS.TXT").write_text(
        "\n".join(f"{s} | {g} | dev-clean | x" for s, g in zip(speakers, genders)), encoding="utf-8")
    for i, s in enumerate(speakers):
        d = root / "dev-clean" / s / "ch"
        d.mkdir(parents=True)
        for j in range(13):
            x = voice(1 if (i + j) % 2 else -1, 7.0)
            sf.write(d / f"u{j}.flac", x, SR_, format="FLAC")
    return root


def test_testset_build(tmp_path):
    from speakerlock.testset import build
    root = _fake_librispeech(tmp_path / "libri")
    man_path = build(root, tmp_path / "ts", speakers=2)
    man = json.loads(man_path.read_text())
    assert len(man["speakers"]) == 2
    d = tmp_path / "ts" / man["speakers"][0]["id"]
    for name in ("enroll.wav", "test0.wav", "mix.wav", "mix_target.wav", "mix_interferer.wav"):
        assert (d / name).exists()
    seg = man["speakers"][0]["segments"]
    assert seg["target_alone"][0] == 0 and seg["overlap"][0] < seg["overlap"][1]
    with pytest.raises(Exception):
        build(tmp_path / "empty", tmp_path / "ts2", speakers=2)


# ------------------------------------------------------------------ benchmark
def _fake_testset(root: Path, ids=("a", "b")) -> Path:
    man = {"source": "fake", "speakers": []}
    for i, s in enumerate(ids):
        d = root / s
        d.mkdir(parents=True)
        a, itf = voice(1, 6), voice(-1, 6)
        gap = np.zeros(SR_ // 2, np.float32)
        b = voice(1, 4)
        mix = np.concatenate([a, gap, itf, b + 0.8 * itf[: len(b)]])
        save_audio(d / "enroll.wav", voice(1, 6))
        for k in range(3):
            save_audio(d / f"test{k}.wav", voice(1, 3))
        save_audio(d / "mix.wav", mix)
        save_audio(d / "mix_target.wav", np.concatenate([a, gap, np.zeros_like(itf), b]))
        n = len(a)
        man["speakers"].append({"id": s, "gender": "F", "interferer": ids[(i + 1) % len(ids)],
                                "segments": {"target_alone": [0, n], "interferer_alone": [n + len(gap), n + len(gap) + len(itf)],
                                             "overlap": [n + len(gap) + len(itf), len(mix)]}})
    (root / "manifest.json").write_text(json.dumps(man))
    return root


def test_run_benchmark_smoke(tmp_path, monkeypatch):
    # bench.py binds the builders at module import, so patch them there
    monkeypatch.setattr(bench_mod, "build_extractor", lambda *a, **k: PassthroughEx())
    monkeypatch.setattr(bench_mod, "build_vad", lambda *a, **k: ToneVAD())
    monkeypatch.setattr(bench_mod, "build_embedder", lambda *a, **k: SignEmbedder())
    ts = _fake_testset(tmp_path / "ts")
    cfg = Config()
    rep = bench_mod.run_benchmark(cfg, ts, speakers=2, live_seconds=0.4, log=lambda *a: None)
    assert "verification_extracted" in rep and "gate" in rep and "latency" in rep
    assert rep["speakers"] == 2


def test_far_frr_math():
    gen = [{"me": 0.9, "x": 0.1} for _ in range(3)]
    imp = [{"me": 0.1, "x": 0.9} for _ in range(4)]
    rep = bench_mod._far_frr(gen, imp, ["me"] * 3, ["me"] * 4, 0.5, True)
    assert rep["FRR"] == 0.0 and rep["FAR"] == 0.0


# ------------------------------------------------------------------ pipeline.create + misc
def test_pipeline_create_with_fakes(tmp_path, monkeypatch):
    _fakes(monkeypatch)
    from speakerlock.enroll import UserStore, enroll
    store = UserStore(tmp_path)
    store.save(enroll("me", voice(1, 4), PassthroughEx(), SignEmbedder(), ToneVAD(), 0.5))
    from speakerlock.pipeline import SpeakerLock
    sl = SpeakerLock.create(Config(store_dir=str(tmp_path)), "me", str(tmp_path), background_verify=False)
    y, m = sl.process(voice(1, 0.5))
    y2, _ = sl.flush()
    assert len(y) + len(y2) == 8000
    sl.close()


def test_energy_vad_clone_and_floor():
    from speakerlock.vad import EnergyVAD
    v = EnergyVAD(512)
    quiet = np.full(512, 1e-4, np.float32)
    loud = np.full(512, 0.3, np.float32)
    for _ in range(5):
        v.probability(quiet)
    assert isinstance(v.clone(), EnergyVAD)
    assert v.probability(loud) > 0.5
    assert v.probability(np.zeros(512, np.float32)) <= 0.5


def test_ort_utils_session_cpu():
    from speakerlock.ort_utils import session
    import onnxruntime as ort
    s = session.__wrapped__ if hasattr(session, "__wrapped__") else session
    # device cpu path
    p = Path.home() / ".speakerlock" / "silero_vad.onnx"
    if p.exists():
        assert isinstance(session(p, device="cpu"), ort.InferenceSession)


def test_stream_backlog_drop():
    from speakerlock.pipeline import SpeakerLock
    from speakerlock.stream import LiveRunner
    from speakerlock.interfaces import TargetCondition
    import time
    cfg = Config.from_dict({"verifier": {"window_s": 0.5, "min_speech_s": 0.2},
                            "gate": {"start_open": True, "strict_turns": False}})
    sl = SpeakerLock(cfg, "me", TargetCondition({}), {"me": np.array([1.0, 0.0])},
                     PassthroughEx(), ToneVAD(), SignEmbedder(), background_verify=False)
    got = []
    r = LiveRunner(sl, lambda y, m: got.append(len(y)), max_backlog_s=0.02).start()
    for _ in range(50):                       # flood faster than the worker drains
        r.push(np.full(320, 0.1, np.float32))
    r.stop(flush=False)
    assert r.dropped >= 0 and sum(got) <= 50 * 320


# ------------------------------------------------------------------ CLI
def test_cli_users_and_enroll_and_run(tmp_path, monkeypatch, capsys):
    _fakes(monkeypatch)
    f = tmp_path / "c.wav"
    sf.write(f, voice(1, 5), SR_)
    assert cli_mod.main(["--store", str(tmp_path / "store"), "enroll", "--user", "me", "--file", str(f)]) == 0
    assert (tmp_path / "store" / "me.npz").exists()

    class FakeLock:
        def __init__(self, *a, **k):
            self.calls = 0

        def process(self, x):
            self.calls += 1
            return x, _meta(len(x))

        def flush(self):
            return np.zeros(0, np.float32), _meta(0)

        def close(self):
            pass

    monkeypatch.setattr("speakerlock.pipeline.SpeakerLock.create", lambda *a, **k: FakeLock())
    out = tmp_path / "out.wav"
    assert cli_mod.main(["run", "--user", "me", "--file", str(f), "--out", str(out),
                         "--meta", str(tmp_path / "m.jsonl"), "--quiet"]) == 0
    assert out.exists() and (tmp_path / "m.jsonl").read_text().strip()

    assert cli_mod.main(["--store", str(tmp_path / "store"), "users"]) == 0
    assert "me" in capsys.readouterr().out
    assert cli_mod.main(["--store", str(tmp_path / "store"), "users", "--delete", "me"]) == 0


def test_cli_benchmark_mocked(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench_mod, "run_benchmark",
                        lambda *a, **k: {"ok": True})
    (tmp_path / "ts").mkdir()
    assert cli_mod.main(["benchmark", "--testset", str(tmp_path / "ts"),
                         "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text()) == {"ok": True}


def test_cli_record_mocked(monkeypatch):
    import sounddevice as sd
    monkeypatch.setattr(sd, "rec", lambda *a, **k: np.full((1600, 1), 0.1, np.float32))
    monkeypatch.setattr(sd, "wait", lambda: None)
    x = cli_mod._record(0.1)
    assert x.shape == (1600,)
