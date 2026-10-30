"""Enrollment: record 5-10 short phrases and store an averaged speaker embedding.

Usage: python enroll.py [--phrases 8] [--seconds 4] [--config config.yaml]
Writes profile/voice.npy (centroid) and profile/enroll_embeddings.npy (per-phrase, for calibrate.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from voice.config import load_config
from voice.dsp import rms_db
from voice.logging_utils import setup_logging
from voice.speaker import average_embedding, build_embedder, cosine_similarity

PHRASES = [
    "Hey assistant, what's the weather like today?",
    "Set a timer for ten minutes, please.",
    "Remind me to call my mother this evening.",
    "The quick brown fox jumps over the lazy dog.",
    "Can you summarise the latest news for me?",
    "Play some relaxing music in the living room.",
    "How long does it take to boil an egg?",
    "Turn the lights off when I leave the house.",
    "Translate good morning into Spanish.",
    "What's on my calendar for tomorrow afternoon?",
]


STARTUP_DISCARD_S = 0.15  # Windows drivers often emit a full-scale click when a capture stream opens


def record(seconds: float, rate: int, device) -> np.ndarray:
    import sounddevice as sd
    from voice.audio_io import _resolve_device
    skip = int(STARTUP_DISCARD_S * rate)
    audio = sd.rec(int(seconds * rate) + skip, samplerate=rate, channels=1, dtype="float32",
                   device=_resolve_device(device, "input"))
    sd.wait()
    return audio[skip:, 0]


def check_take(audio: np.ndarray) -> tuple[bool, str, float]:
    """Quality gate for an enrollment take: (ok, problem, level_dbfs).
    Clipping = more than 0.5% of samples saturated; a stray click (~0.2%) doesn't count."""
    level = rms_db(audio)
    if level < -45:
        return False, "too_quiet", level
    if audio.size and np.mean(np.abs(audio) > 0.99) > 0.005:
        return False, "clipping", level
    return True, "", level


def save_profile(embs: list[np.ndarray], cfg) -> dict:
    """Write the centroid + per-take embeddings + metadata. Returns the metadata."""
    centroid = average_embedding(embs)
    sims = [cosine_similarity(e, centroid) for e in embs]
    out = Path(cfg.speaker.profile_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, centroid)
    np.save(out.with_name("enroll_embeddings.npy"), np.stack(embs))
    meta = {"backend": cfg.speaker.backend, "phrases": len(embs), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "self_similarity_mean": float(np.mean(sims)), "self_similarity_min": float(np.min(sims))}
    out.with_name("profile.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrases", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    n = max(5, min(10, args.phrases))
    rate = cfg.audio.sample_rate

    print(f"Loading speaker model ({cfg.speaker.backend})...")
    embedder = build_embedder(cfg.speaker.backend)
    embs = []
    print(f"\nYou'll read {n} phrases, {args.seconds:.0f}s each. Speak naturally at your usual distance from the mic.\n")
    while len(embs) < n:
        i = len(embs)
        input(f"[{i + 1}/{n}] Press Enter, then say:  \"{PHRASES[i % len(PHRASES)]}\"")
        print("  recording...", end="", flush=True)
        audio = record(args.seconds, rate, cfg.audio.input_device)
        ok, problem, level = check_take(audio)
        print(f" done (level {level:.1f} dBFS)")
        if problem == "too_quiet":
            print("  Too quiet - check the mic / speak louder. Retrying this phrase.")
        elif problem == "clipping":
            print("  Clipping detected - move back from the mic a bit. Retrying this phrase.")
        if ok:
            embs.append(embedder.embed(audio))

    meta = save_profile(embs, cfg)
    print(f"\nSaved {cfg.speaker.profile_path}  (self-similarity mean {meta['self_similarity_mean']:.3f}, "
          f"min {meta['self_similarity_min']:.3f})")
    if meta["self_similarity_min"] < 0.7:
        print("Warning: enrollment clips are inconsistent (noise? different distances?). Consider re-running.")
    print("Next: python calibrate.py   (suggests a threshold)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
