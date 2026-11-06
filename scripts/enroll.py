"""Owner voice enrollment.

Records N phrases from the microphone, embeds them with the configured ONNX
speaker model, averages into an L2-normalised centroid, and saves it to
``profile/owner.npy`` (or the path set in firewall.yaml).

Usage
-----
  python scripts/enroll.py                      # 8 phrases × 4 s from mic
  python scripts/enroll.py --phrases 5          # 5 phrases
  python scripts/enroll.py --files a.wav b.wav  # use existing WAV files instead of mic
  python scripts/enroll.py --out profile/me.npy # custom output path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _record_phrase(phrase_num: int, seconds: float, sample_rate: int = 16000):
    """Record one phrase from the default input device using sounddevice."""
    import numpy as np
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] sounddevice is not installed.  pip install sounddevice", file=sys.stderr)
        raise

    frames = int(seconds * sample_rate)
    print(f"  Phrase {phrase_num}: speak now for {seconds:.1f} s…", end=" ", flush=True)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("done.")
    return audio.ravel()


def _load_wav(path: Path, target_sr: int = 16000):
    """Load a WAV file → float32 16 kHz mono."""
    import wave
    import numpy as np
    import math

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio /= float(np.iinfo(dtype).max)

    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(math.ceil(len(audio) * ratio))
        old_idx = np.arange(new_len) / ratio
        audio = np.interp(old_idx, np.arange(len(audio)), audio).astype(np.float32)

    return audio


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Enroll owner voice profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="firewall.yaml",
                    help="firewall.yaml to read speaker model and profile path from")
    ap.add_argument("--phrases", type=int, default=8,
                    help="number of phrases to record (default: 8)")
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="seconds per phrase (default: 4.0)")
    ap.add_argument("--files", nargs="+", metavar="WAV",
                    help="use existing WAV files instead of recording from mic")
    ap.add_argument("--out", default=None,
                    help="output profile path (default: taken from firewall.yaml)")
    args = ap.parse_args()

    # Load config
    try:
        from voice_firewall._config import FirewallConfig
        cfg = FirewallConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"[ERROR] Config not found: '{args.config}'.  Run from the VoiceFirewall directory.",
              file=sys.stderr)
        return 1

    out_path = args.out or cfg.guards.speaker.profile_path

    # Build embedder
    from voice_firewall.speaker.embedder import build_embedder
    print(f"Loading speaker model ({cfg.guards.speaker.onnx_model})…")
    embedder = build_embedder(model_name=cfg.guards.speaker.onnx_model)

    # Collect audio
    clips = []
    if args.files:
        for fname in args.files:
            p = Path(fname)
            if not p.exists():
                print(f"[WARN] File not found: '{p}'", file=sys.stderr)
                continue
            try:
                audio = _load_wav(p)
                clips.append(audio)
                print(f"  Loaded '{p.name}'  ({len(audio)/16000:.1f} s)")
            except Exception as exc:
                print(f"[WARN] Could not load '{p}': {exc}", file=sys.stderr)
    else:
        print(f"\nEnrolling owner voice — {args.phrases} phrase(s) × {args.seconds:.1f} s each")
        print("Speak naturally; avoid silence at the edges.\n")
        for i in range(1, args.phrases + 1):
            try:
                audio = _record_phrase(i, args.seconds)
                clips.append(audio)
            except Exception as exc:
                print(f"[ERROR] Recording failed: {exc}", file=sys.stderr)
                return 1

    if not clips:
        print("[ERROR] No audio clips collected.  Aborting.", file=sys.stderr)
        return 1

    # Embed each clip
    print(f"\nEmbedding {len(clips)} clip(s)…")
    embeddings = []
    for i, audio in enumerate(clips, 1):
        try:
            emb = embedder.embed(audio, sample_rate=16000)
            embeddings.append(emb)
            print(f"  Clip {i}: OK  ({len(audio)/16000:.1f} s)")
        except Exception as exc:
            print(f"  Clip {i}: FAILED ({exc})", file=sys.stderr)

    if not embeddings:
        print("[ERROR] All clips failed to embed.  Aborting.", file=sys.stderr)
        return 1

    # Save centroid profile
    from voice_firewall.speaker.gate import save_profile
    centroid = save_profile(embeddings, out_path)

    print(f"\n[OK] Profile saved to '{out_path}'")
    print(f"     {len(embeddings)} embedding(s) averaged into a 256-dim centroid.")
    print("\nNext step:  run observe mode for a few sessions, then calibrate thresholds:")
    print("  python scripts/calibrate.py --owner audio/owner/ --nonowner audio/strangers/ --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
