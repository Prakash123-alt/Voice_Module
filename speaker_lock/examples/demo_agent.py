"""One-command demo: an agent that only hears the enrolled speaker.

    python examples/demo_agent.py [--config config.example.yaml]

Builds a 3-part scene from tests/data (target speech -> interferer speech -> overlap + noise),
streams it through SpeakerLock, transcribes whatever passes the gate with faster-whisper, and
prints the mock agent's reply. The interferer's words should never reach the transcript.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speakerlock import Config  # noqa: E402
from speakerlock.audio_io import load_audio  # noqa: E402
from speakerlock.embedder import build_embedder  # noqa: E402
from speakerlock.enroll import enroll  # noqa: E402
from speakerlock.extractor import build_extractor  # noqa: E402
from speakerlock.integrations.whisper_adapter import WhisperAdapter  # noqa: E402
from speakerlock.pipeline import SpeakerLock  # noqa: E402
from speakerlock.vad import build_vad  # noqa: E402
from speakerlock.config import SAMPLE_RATE as SR  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"


def reply(text: str) -> str:
    return f"I heard you say: {text!r} (mock reply - plug in your LLM here)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--whisper-model", default="tiny.en")
    ap.add_argument("--no-stt", action="store_true", help="skip transcription, only show gate decisions")
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config) if a.config else Config()
    ex = build_extractor(cfg.extractor, cfg.device)
    emb = build_embedder(cfg.verifier, cfg.device)
    vad = build_vad(cfg.vad, cfg.frame_samples, cfg.device)

    print("enrolling 'me' from tests/data/target_enroll.wav ...")
    me = enroll("me", load_audio(DATA / "target_enroll.wav"), ex, emb, vad, cfg.vad.threshold)

    tgt, itf = load_audio(DATA / "target_query.wav"), load_audio(DATA / "interferer.wav")
    rng = np.random.default_rng(0)
    gap = np.zeros(SR // 2, np.float32)
    n = len(tgt)
    scene = np.concatenate([tgt, gap, itf, tgt + 0.8 * itf[:n]])
    scene = scene + rng.normal(0, 0.003, len(scene)).astype(np.float32)
    print(f"scene: {len(scene) / SR:.1f}s  [target alone] [interferer alone] [target+interferer overlap] + noise")

    def on_unverified(info):
        print(f"\n  >> on_unverified fired: best match was {info['best_user']!r} "
              f"(score {info['score']:.2f}, {info['failures']} failing windows)")

    lock = SpeakerLock(cfg, "me", me.condition, {"me": me.dvector}, ex, vad, emb,
                       on_unverified=on_unverified)
    stt = None
    if not a.no_stt:
        try:
            from speakerlock.integrations.whisper_adapter import faster_whisper_transcriber
            stt = WhisperAdapter(faster_whisper_transcriber(a.whisper_model))
        except ImportError:
            print("faster-whisper not installed - running with --no-stt (pip install faster-whisper)")
    print("\nstreaming ...\n")
    heard = []
    for i in range(0, len(scene), cfg.frame_samples):
        y, m = lock.process(scene[i:i + cfg.frame_samples])
        if stt is not None:
            for text in stt.feed(y, m):
                heard.append(text)
                print(f"  STT (verified speech only): {text!r}")
                print(f"  agent: {reply(text)}")
        print(f"\r  t={m.start_sample / SR:5.1f}s  {'VERIFIED' if m.verified else 'blocked '} "
              f"score={m.score if m.score is not None else '-'}  vad={m.vad:.2f}   ", end="", flush=True)
    y, m = lock.flush()
    if stt is not None:
        for text in stt.feed(y, m) + stt.flush():
            heard.append(text)
            print(f"\n  STT: {text!r}\n  agent: {reply(text)}")
    lock.close()
    print("\n\ndone. Only segments marked VERIFIED reached the STT/agent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
