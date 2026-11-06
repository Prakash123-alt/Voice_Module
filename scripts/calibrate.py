"""Threshold calibration — computes FAR/FRR and writes thresholds to firewall.yaml.

Usage
-----
  # From audio files in two directories:
  python scripts/calibrate.py --owner audio/owner/ --nonowner audio/strangers/ --write

  # From an observe-mode labeled score log:
  python scripts/calibrate.py --log scorelog.jsonl --write

  # Dry-run (print report, do not write):
  python scripts/calibrate.py --owner audio/owner/ --nonowner audio/strangers/

Arguments
---------
--owner PATH [PATH ...]   WAV files or directory containing owner recordings.
--nonowner PATH [PATH ...]  WAV files or directory containing non-owner recordings.
--log PATH                Labeled JSONL score log (label: "owner" / "nonowner").
--config PATH             firewall.yaml to read and optionally update (default: firewall.yaml).
--write                   Write recommended thresholds to the config file.
--strict-margin FLOAT     Extra margin added to threshold for strict_threshold (default: 0.05).
--target-far FLOAT        Target FAR for the recommended threshold (default: 0.05 = 5%).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _collect_wav_files(paths: list[str]) -> list[Path]:
    """Expand directories and return all .wav files."""
    result = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            result.extend(sorted(pp.glob("*.wav")))
        elif pp.suffix.lower() == ".wav" and pp.exists():
            result.append(pp)
        else:
            print(f"[WARN] Skipping '{p}': not a .wav file or directory", file=sys.stderr)
    return result


def _score_wav_files(wav_files: list[Path], embedder, profile) -> list[float]:
    """Embed each WAV file and compute cosine similarity vs the enrolled profile."""
    import numpy as np

    from voice_firewall.speaker.gate import CosineSpeakerGate

    scores = []
    for wav_path in wav_files:
        try:
            audio = _load_wav(wav_path)
            emb = embedder.embed(audio, sample_rate=16000)
            score = float(np.dot(
                emb.astype(np.float64),
                profile.astype(np.float64)
            ))
            scores.append(score)
        except Exception as exc:
            print(f"[WARN] Could not score '{wav_path}': {exc}", file=sys.stderr)
    return scores


def _load_wav(path: Path):
    """Load a WAV file and return float32 16 kHz mono audio."""
    import wave
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    # Normalise to [-1, 1]
    maxval = float(np.iinfo(dtype).max)
    audio /= maxval

    # Resample to 16 kHz if needed
    if sr != 16000:
        import math
        ratio = 16000 / sr
        new_len = int(math.ceil(len(audio) * ratio))
        old_idx = np.arange(new_len) / ratio
        audio = np.interp(old_idx, np.arange(len(audio)), audio).astype(np.float32)

    return audio


def _write_thresholds_to_yaml(config_path: str, threshold: float, strict_threshold: float) -> None:
    """Update guards.speaker.threshold and strict_threshold in a YAML config file."""
    import re

    text = Path(config_path).read_text(encoding="utf-8")

    def replace_field(text: str, key: str, value: float) -> str:
        pattern = rf"(^\s*{key}\s*:\s*)(?:null|[\d.]+)"
        replacement = rf"\g<1>{value:.4f}"
        new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if count == 0:
            print(f"[WARN] Field '{key}' not found in {config_path}; skipping", file=sys.stderr)
        return new_text

    text = replace_field(text, "threshold", threshold)
    text = replace_field(text, "strict_threshold", strict_threshold)
    # Also disable observe_mode (calibrated → switch to enforcement mode)
    text = re.sub(
        r"(^\s*observe_mode\s*:\s*)true",
        r"\g<1>false",
        text, flags=re.MULTILINE,
    )

    Path(config_path).write_text(text, encoding="utf-8")
    print(f"[OK] Wrote thresholds to '{config_path}'")
    print(f"     threshold={threshold:.4f}  strict_threshold={strict_threshold:.4f}")
    print("     observe_mode → false (enforcement mode active)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate speaker thresholds from recorded audio or score log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--owner", nargs="+", metavar="PATH",
                    help="owner audio clips (.wav) or directory")
    ap.add_argument("--nonowner", nargs="+", metavar="PATH",
                    help="non-owner audio clips (.wav) or directory")
    ap.add_argument("--log", metavar="JSONL",
                    help="labeled JSONL score log (label: 'owner' / 'nonowner')")
    ap.add_argument("--config", default="firewall.yaml",
                    help="path to firewall.yaml (default: firewall.yaml)")
    ap.add_argument("--write", action="store_true",
                    help="write thresholds to --config (default: dry-run)")
    ap.add_argument("--strict-margin", type=float, default=0.05,
                    help="extra margin for strict_threshold above threshold (default: 0.05)")
    ap.add_argument("--target-far", type=float, default=0.05,
                    help="target FAR for the recommended threshold (default: 0.05 = 5%%)")
    args = ap.parse_args()

    if not args.owner and not args.nonowner and not args.log:
        print("[ERROR] Provide --owner + --nonowner directories/files, or --log scorelog.jsonl",
              file=sys.stderr)
        ap.print_help(sys.stderr)
        return 1

    import numpy as np
    from voice_firewall.calibration import CalibrationResult, ScoreLog, compute_calibration

    owner_scores: list[float] = []
    nonowner_scores: list[float] = []

    # --- Source 1: audio files ---
    if args.owner or args.nonowner:
        # Load config and profile
        try:
            from voice_firewall._config import FirewallConfig
            cfg = FirewallConfig.from_yaml(args.config)
        except FileNotFoundError:
            print(f"[ERROR] Config not found: '{args.config}'", file=sys.stderr)
            return 1

        from voice_firewall.speaker.embedder import build_embedder
        from voice_firewall.speaker.gate import load_profile

        profile = load_profile(cfg.guards.speaker.profile_path)
        if profile is None:
            print(
                f"[ERROR] No enrolled profile at '{cfg.guards.speaker.profile_path}'.\n"
                "        Run 'python scripts/enroll.py' first.",
                file=sys.stderr,
            )
            return 1

        embedder = build_embedder(model_name=cfg.guards.speaker.onnx_model)

        if args.owner:
            wav_files = _collect_wav_files(args.owner)
            if not wav_files:
                print("[WARN] No owner WAV files found", file=sys.stderr)
            else:
                print(f"Scoring {len(wav_files)} owner file(s)…")
                owner_scores.extend(_score_wav_files(wav_files, embedder, profile))

        if args.nonowner:
            wav_files = _collect_wav_files(args.nonowner)
            if not wav_files:
                print("[WARN] No non-owner WAV files found", file=sys.stderr)
            else:
                print(f"Scoring {len(wav_files)} non-owner file(s)…")
                nonowner_scores.extend(_score_wav_files(wav_files, embedder, profile))

    # --- Source 2: labeled JSONL score log ---
    if args.log:
        log_owner, log_nonowner = ScoreLog.read(args.log)
        owner_scores.extend(log_owner.tolist())
        nonowner_scores.extend(log_nonowner.tolist())
        print(f"Read {len(log_owner)} owner + {len(log_nonowner)} non-owner scores from log")

    if not owner_scores:
        print("[ERROR] No owner scores collected. Check --owner paths.", file=sys.stderr)
        return 1
    if not nonowner_scores:
        print("[ERROR] No non-owner scores collected. Check --nonowner paths.", file=sys.stderr)
        return 1

    # --- Calibrate ---
    try:
        result = compute_calibration(
            np.array(owner_scores, dtype=np.float64),
            np.array(nonowner_scores, dtype=np.float64),
            strict_margin=args.strict_margin,
        )
    except ValueError as exc:
        print(f"[ERROR] Calibration failed: {exc}", file=sys.stderr)
        return 1

    print()
    print(result.report())

    # --- Write ---
    if args.write:
        _write_thresholds_to_yaml(args.config, result.threshold, result.strict_threshold)
    else:
        print("\n[DRY-RUN] Pass --write to apply these thresholds to the config file.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
