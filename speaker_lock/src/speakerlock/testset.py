"""Build a small, reproducible evaluation set from LibriSpeech (CC BY 4.0).

For each of N speakers (alternating gender): a 5 s enrollment clip, 3 genuine test utterances, and one
mixture laid out as

    [target alone] [0.5 s gap] [interferer alone] [target + interferer, 0 dB]   + pink noise (20 dB SNR)

with the clean target / interferer tracks saved as ground truth.
Get the data:  https://www.openslr.org/resources/12/dev-clean.tar.gz  (337 MB)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .audio_io import load_audio, save_audio
from .config import SAMPLE_RATE

SR = SAMPLE_RATE


def _speakers(root: Path) -> list[tuple[str, str]]:
    genders = {}
    for line in (root / "SPEAKERS.TXT").read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^\s*(\d+)\s*\|\s*([MF])\s*\|\s*dev-clean", line)
        if m:
            genders[m[1]] = m[2]
    present = sorted(p.name for p in (root / "dev-clean").iterdir() if p.is_dir())
    f = [s for s in present if genders.get(s) == "F"]
    m_ = [s for s in present if genders.get(s) == "M"]
    out = []
    for a, b in zip(f, m_):
        out += [(a, "F"), (b, "M")]
    return out


def _utterances(root: Path, spk: str) -> list[Path]:
    return sorted((root / "dev-clean" / spk).rglob("*.flac"))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.arange(spec.shape[0]), 1))
    x = np.fft.irfft(spec, n)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def build(librispeech_root: str | Path, out_dir: str | Path, speakers: int = 8, seed: int = 0) -> Path:
    root, out = Path(librispeech_root).expanduser(), Path(out_dir).expanduser()
    rng = np.random.default_rng(seed)
    spk = _speakers(root)[:speakers]
    if len(spk) < 2:
        raise RuntimeError(f"need LibriSpeech dev-clean under {root}")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": "LibriSpeech dev-clean (CC BY 4.0)", "speakers": []}
    audio: dict[str, list[np.ndarray]] = {}
    for s, g in spk:
        utts = [load_audio(p) for p in _utterances(root, s)[:12]]
        utts.sort(key=len, reverse=True)
        audio[s] = utts
    for i, (s, g) in enumerate(spk):
        d = out / s
        d.mkdir(exist_ok=True)
        utts = audio[s]
        enroll = utts[0][: 5 * SR]
        save_audio(d / "enroll.wav", enroll)
        tests = [u[: 8 * SR] for u in utts[1:4]]
        for k, t in enumerate(tests):
            save_audio(d / f"test{k}.wav", t)
        other = spk[(i + 1) % len(spk)][0]                 # adjacent speaker = opposite gender
        a, b, itf = utts[4][: 6 * SR], utts[5][: 6 * SR], audio[other][6][: 6 * SR]
        itf = itf * (_rms(a) / _rms(itf))                    # 0 dB SIR
        gap = np.zeros(SR // 2, np.float32)
        n = min(len(b), len(itf))
        target_track = np.concatenate([a, gap, np.zeros_like(itf), b[:n]])
        itf_track = np.concatenate([np.zeros_like(a), gap, itf, itf[:n]])
        clean = target_track + itf_track
        noise = pink_noise(len(clean), rng) * _rms(a) / (10 ** (20 / 20))
        mix = (clean + noise).astype(np.float32)
        peak = max(1.0, float(np.abs(mix).max()) / 0.95)
        for name, x in (("mix", mix), ("mix_target", target_track), ("mix_interferer", itf_track)):
            save_audio(d / f"{name}.wav", x / peak)
        manifest["speakers"].append({
            "id": s, "gender": g, "interferer": other,
            "segments": {"target_alone": [0, len(a)], "interferer_alone": [len(a) + len(gap), len(a) + len(gap) + len(itf)],
                         "overlap": [len(a) + len(gap) + len(itf), len(mix)]},
        })
    p = out / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return p
