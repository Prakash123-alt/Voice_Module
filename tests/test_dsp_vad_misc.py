import logging

import numpy as np
import pytest

from voice.aec import DelayLine, GateAEC
from voice.config import load_config
from voice.dsp import FrameBuffer, estimate_delay_samples, from_wav_bytes, resample, to_wav_bytes
from voice.logging_utils import _KVFormatter, redact
from voice.vad import Segmenter


def run_seg(seg, probs):
    evs = []
    for i, p in enumerate(probs):
        ev = seg.push(np.full(320, i, dtype=np.float32), p)
        if ev:
            evs.append(ev)
    return evs


def test_segmenter_preroll_hangover_and_debounce():
    seg = Segmenter(frame_ms=20, threshold=0.5, min_speech_ms=60, hangover_ms=100, pad_ms=40)
    probs = [0] * 10 + [1] * 2 + [0] * 3 + [1] * 10 + [0] * 5
    evs = run_seg(seg, probs)
    kinds = [e.kind for e in evs]
    assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("start") == 1  # 2-frame blip ignored
    end = evs[-1]
    # speech starts at frame 15, confirmed at 17; pre-roll adds 2 frames (40 ms) before 15
    assert end.audio[0] == 13
    assert end.speech_ms == 10 * 20
    assert end.audio.shape[0] == (2 + 10 + 5) * 320  # pad + speech + hangover


def test_segmenter_short_pause_does_not_split():
    seg = Segmenter(frame_ms=20, min_speech_ms=40, hangover_ms=200, pad_ms=0)
    probs = [1] * 10 + [0] * 5 + [1] * 10 + [0] * 10
    evs = [e for e in run_seg(seg, probs) if e.kind in ("start", "end")]
    assert [e.kind for e in evs] == ["start", "end"]


def test_segmenter_max_length_cut():
    seg = Segmenter(frame_ms=20, min_speech_ms=20, hangover_ms=500, max_segment_s=0.2)
    evs = run_seg(seg, [1] * 30)
    assert any(e.kind == "end" for e in evs)


def test_estimate_delay_finds_lag():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000).astype(np.float32)
    lag = 1600  # 100 ms
    mic = np.concatenate([np.zeros(lag, np.float32), 0.3 * ref[:-lag]]) + 0.01 * rng.standard_normal(16000).astype(np.float32)
    got, conf = estimate_delay_samples(ref, mic, max_lag=6400)
    assert abs(got - lag) <= 2 and conf > 4


def test_delay_line():
    dl = DelayLine(4, delay_frames=2, max_frames=5)
    outs = [dl.push(np.full(4, i, np.float32))[0] for i in range(5)]
    assert outs == [0, 0, 0, 1, 2]


def test_gate_aec_mutes_while_reference_active():
    g = GateAEC(320, tail_frames=2)
    mic = np.ones(320, np.float32)
    assert not g.process(mic, np.ones(320, np.float32)).any()
    assert not g.process(mic, None).any()      # tail
    assert g.process(mic, None).all()           # tail expired


def test_resample_length_and_wav_roundtrip():
    x = np.sin(np.linspace(0, 100, 48000)).astype(np.float32) * 0.5
    y = resample(x, 48000, 16000)
    assert y.shape[0] == 16000
    back, rate = from_wav_bytes(to_wav_bytes(y, 16000))
    assert rate == 16000 and np.allclose(back, y, atol=1e-3)


def test_framebuffer():
    fb = FrameBuffer(3)
    assert [f.tolist() for f in fb.push(np.arange(7, dtype=np.float32))] == [[0, 1, 2], [3, 4, 5]]
    assert fb.flush().tolist() == [6, 0, 0]


def test_redaction():
    key = "sk-or-v1-MOCK-TEST-KEY-123456789"
    assert key not in redact(f"key={key}")
    assert "abc.def" not in redact("Authorization: Bearer abc.def")
    rec = logging.LogRecord("x", logging.INFO, "", 0, "hello", (), None)
    rec.kv = {"auth": f"Bearer {key}"}
    assert key not in _KVFormatter(False).format(rec)


def test_load_config_defaults_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-zzz")
    p = tmp_path / "c.yaml"
    p.write_text("speaker:\n  threshold: 0.7\naudio:\n  input_device: null\naec:\n  delay_ms: auto\n")
    cfg = load_config(p, env_file=tmp_path / "none.env")
    assert cfg.speaker.threshold == 0.7 and cfg.speaker.min_segment_ms == 400
    assert cfg.audio.frame_samples == 320 and cfg.aec.delay_ms == "auto"
    assert cfg.openrouter.api_key == "sk-or-v1-zzz"
    assert "sk-or" not in repr(cfg.openrouter)


def test_repo_config_loads():
    cfg = load_config("config.yaml", env_file="nonexistent.env")
    assert cfg.llm.model and 100 <= cfg.barge_in.min_speech_ms <= 500 and 0.3 <= cfg.speaker.threshold <= 0.95


class _Seg:
    def __init__(self, text, logprob=-0.2, compression=1.2, no_speech=0.05):
        self.text, self.avg_logprob = text, logprob
        self.compression_ratio, self.no_speech_prob = compression, no_speech


def test_stt_segments_drop_hallucinations():
    from voice.stt import _segments_text
    assert _segments_text([_Seg("Hello there."), _Seg(" General Kenobi.")]) == "Hello there. General Kenobi."
    # classic repetition hallucination ("Low, low, low..."): high compression ratio
    assert _segments_text([_Seg("Low, low, low, low, low, low.", compression=4.8)]) == ""
    # mumbling/noise: very low logprob
    assert _segments_text([_Seg("uh...", logprob=-1.4)]) == ""
    # silence classified as speech by VAD: high no_speech + mediocre logprob
    assert _segments_text([_Seg("Thank you.", logprob=-0.7, no_speech=0.9)]) == ""
    # confident segment on real speech is kept even with moderate no_speech
    assert _segments_text([_Seg("Yes.", logprob=-0.1, no_speech=0.9)]) == "Yes."


def test_segmenter_speech_only_window_excludes_preroll():
    seg = Segmenter(frame_ms=20, threshold=0.5, min_speech_ms=60, hangover_ms=100, pad_ms=100)
    speech = np.full(320, 0.3, dtype=np.float32)
    silence = np.zeros(320, dtype=np.float32)
    for _ in range(6):
        assert seg.push(silence, 0.0) is None
    ev = None
    for _ in range(4):
        ev = seg.push(speech, 0.9) or ev
    assert ev is not None and ev.kind in ("start", "speech")
    speech_only = seg.current_speech_audio()
    assert np.all(speech_only > 0.1)              # no silence pre-roll leaked in
    assert speech_only.shape[0] == 4 * 320
    assert seg.current_audio().shape[0] > speech_only.shape[0]  # full window still carries the pad
