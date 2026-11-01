"""Generate the small speech clips used by the unit tests (tests/data/*.wav, 16 kHz int16).

Uses three Piper voices (lessac = target, ryan = interferer, amy = third speaker). Run once:
    python scripts/make_test_assets.py --piper-dir ../models
"""
from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

TEXTS = {
    "target_enroll": ("lessac", "Hello, this is my voice. I am recording a short sample so the assistant "
                                "can recognise me, even when other people are talking nearby."),
    "target_query": ("lessac", "Who is the chief minister of Andhra Pradesh? Please answer briefly."),
    "interferer": ("ryan", "The weather report says it will rain all weekend, so bring an umbrella "
                           "and plan your trip to the market for Monday morning."),
    "third": ("amy", "I think the football match was fantastic last night, especially the second half."),
}


def synth(voice_path: Path, text: str) -> np.ndarray:
    from piper import PiperVoice
    from scipy.signal import resample_poly
    voice = PiperVoice.load(str(voice_path))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w)
    buf.seek(0)
    with wave.open(buf) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return resample_poly(a, 16000, sr).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piper-dir", default=str(Path(__file__).resolve().parents[2] / "models"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (spk, text) in TEXTS.items():
        x = synth(Path(a.piper_dir) / f"en_US-{spk}-medium.onnx", text)
        sf.write(out / f"{name}.wav", x, 16000, subtype="PCM_16")
        print(f"{name}.wav  {len(x) / 16000:.1f}s  ({spk})")


if __name__ == "__main__":
    main()
