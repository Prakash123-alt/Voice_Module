"""Suggest a speaker-verification threshold from enrollment data.

Usage:
  python calibrate.py                       # leave-one-out on enrollment embeddings
  python calibrate.py --impostor other.wav  # also score impostor audio (16 kHz mono WAV, or any WAV)
  python calibrate.py --live 3              # record 3 fresh genuine phrases and score them too
  python calibrate.py --write               # write the suggested threshold into config.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from voice.config import load_config
from voice.dsp import from_wav_bytes, resample
from voice.speaker import average_embedding, build_embedder, cosine_similarity, suggest_threshold


def leave_one_out(embs: list[np.ndarray]) -> list[float]:
    """Each enrollment clip vs the centroid of the others (an honest genuine-score estimate)."""
    return [cosine_similarity(e, average_embedding(embs[:i] + embs[i + 1:])) for i, e in enumerate(embs)]


def write_threshold(config_path: Path, value: float) -> bool:
    """Set speaker.threshold in config.yaml in place, keeping comments intact."""
    txt, n_sub = re.subn(r"(^speaker:[ \t]*\r?\n(?:[ \t]+.*\r?\n)*?[ \t]+threshold:[ \t]*)[0-9.]+",
                         rf"\g<1>{value}", config_path.read_text(encoding="utf-8"), count=1, flags=re.M)
    if n_sub:
        config_path.write_text(txt, encoding="utf-8")
    return bool(n_sub)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--impostor", nargs="*", default=[], help="WAV files of other speakers")
    ap.add_argument("--live", type=int, default=0, help="record N fresh phrases from you")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    prof = Path(cfg.speaker.profile_path)
    emb_path = prof.with_name("enroll_embeddings.npy")
    if not emb_path.exists():
        print("No enrollment embeddings found. Run `python enroll.py` first.")
        return 1
    embs = list(np.load(emb_path))
    centroid = np.load(prof)

    genuine = leave_one_out(embs)
    impostor: list[float] = []
    embedder = build_embedder(cfg.speaker.backend) if (args.impostor or args.live) else None
    for f in args.impostor:
        audio, rate = from_wav_bytes(Path(f).read_bytes())
        audio = resample(audio, rate, cfg.audio.sample_rate)
        step = cfg.audio.sample_rate * 3  # 3 s windows
        for s in range(0, max(1, audio.shape[0] - step + 1), step):
            impostor.append(cosine_similarity(embedder.embed(audio[s:s + step]), centroid))
    if args.live:
        from enroll import PHRASES, record
        for i in range(args.live):
            input(f"[live {i + 1}/{args.live}] Press Enter and say: \"{PHRASES[-(i + 1)]}\"")
            genuine.append(cosine_similarity(embedder.embed(record(args.seconds, cfg.audio.sample_rate,
                                                                   cfg.audio.input_device)), centroid))

    rep = suggest_threshold(genuine, impostor or None)
    print(f"Genuine scores : n={len(genuine)} mean={rep['genuine_mean']:.3f} std={rep['genuine_std']:.3f} min={rep['genuine_min']:.3f}")
    if impostor:
        print(f"Impostor scores: n={len(impostor)} mean={rep['impostor_mean']:.3f} max={rep['impostor_max']:.3f}  EER~{rep['eer']:.2%}")
    print(f"Current threshold : {cfg.speaker.threshold}")
    print(f"Suggested threshold: {rep['suggested']}")
    if rep["suggested"] < 0.5:
        print("Note: a very low threshold suggests noisy/inconsistent enrollment - re-enroll in a quiet room.")
    if args.write:
        p = Path(args.config)
        if not write_threshold(p, rep["suggested"]):
            print(f"Could not find speaker.threshold in {p}; set it manually.")
            return 1
        print(f"Wrote threshold {rep['suggested']} to {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
