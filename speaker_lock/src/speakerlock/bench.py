"""Benchmark: latency (p50/p95), real-time factor, verification FAR/FRR, and gate leakage on a test set
built by speakerlock.testset (LibriSpeech mixtures)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .audio_io import load_audio
from .config import SAMPLE_RATE, Config
from .embedder import build_embedder
from .enroll import enroll
from .extractor import build_extractor
from .interfaces import cosine
from .pipeline import SpeakerLock
from .stream import LiveRunner
from .vad import build_vad

SR = SAMPLE_RATE


def _db(num: float, den: float) -> float:
    return round(float(10 * np.log10((num + 1e-12) / (den + 1e-12))), 2)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    s = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    return float(10 * np.log10(np.dot(s, s) / (np.dot(est - s, est - s) + 1e-12)))


def _pct(x, q) -> float:
    return round(float(np.percentile(x, q)), 2) if len(x) else float("nan")


class _Windows:
    """Score 1.6 s windows of VAD-positive speech every hop, exactly as the runtime verifier does."""

    def __init__(self, cfg: Config, vad, emb) -> None:
        self.cfg, self.vad, self.emb = cfg, vad, emb

    def __call__(self, audio: np.ndarray, profiles: dict[str, np.ndarray]) -> list[dict[str, float]]:
        v, F = self.cfg.verifier, self.cfg.frame_samples
        self.vad.reset()
        speech: list[np.ndarray] = []
        out, since = [], 0
        for i in range(0, len(audio) - F + 1, F):
            f = audio[i:i + F]
            if self.vad.probability(f) < self.cfg.vad.threshold:
                continue
            speech.append(f)
            since += F
            win = speech[-int(v.window_s * SR / F):]
            if sum(map(len, win)) >= v.min_speech_s * SR and since >= v.hop_ms * 16:
                since = 0
                e = self.emb.embed(np.concatenate(win))
                out.append({u: cosine(e, p) for u, p in profiles.items()})
        return out


def _far_frr(genuine: list[dict], impostor: list[dict], target_of: list[str], imp_target_of: list[str],
             thr: float, closed_set: bool) -> dict:
    def passes(s: dict, t: str) -> bool:
        return s[t] >= thr and (not closed_set or max(s, key=s.get) == t)
    g = [passes(s, t) for s, t in zip(genuine, target_of)]
    i = [passes(s, t) for s, t in zip(impostor, imp_target_of)]
    gs = np.array([s[t] for s, t in zip(genuine, target_of)])
    is_ = np.array([s[t] for s, t in zip(impostor, imp_target_of)])
    grid = np.linspace(-0.2, 1.0, 1201)
    far = np.array([(is_ >= c).mean() if len(is_) else 0.0 for c in grid])
    frr = np.array([(gs < c).mean() for c in grid])
    k = int(np.argmin(np.abs(far - frr)))
    return {"threshold": thr, "closed_set": closed_set, "genuine_windows": len(g), "impostor_windows": len(i),
            "FRR": round(1 - float(np.mean(g)), 4) if g else None, "FAR": round(float(np.mean(i)), 4) if i else 0.0,
            "EER": round(float((far[k] + frr[k]) / 2), 4), "EER_threshold": round(float(grid[k]), 3),
            "genuine_score_p5_p50": [_pct(gs, 5), _pct(gs, 50)], "impostor_score_p50_p95": [_pct(is_, 50), _pct(is_, 95)]}


def run_benchmark(cfg: Config, testset: Path, speakers: int = 8, live_seconds: float = 20.0,
                  log=print, parts: tuple[str, ...] = ("verification", "gate", "latency")) -> dict:
    man = json.loads((testset / "manifest.json").read_text())
    spk = man["speakers"][:speakers]
    ex = build_extractor(cfg.extractor, cfg.device)
    emb = build_embedder(cfg.verifier, cfg.device)
    vad = build_vad(cfg.vad, cfg.frame_samples, cfg.device)
    ids = [s["id"] for s in spk]
    log(f"enrolling {len(ids)} speakers...")
    users = {s: enroll(s, load_audio(testset / s / "enroll.wav"), ex, emb, vad, cfg.vad.threshold) for s in ids}
    profiles = {s: e.dvector for s, e in users.items()}
    tests = {s: [load_audio(p) for p in sorted((testset / s).glob("test*.wav"))] for s in ids}
    windows = _Windows(cfg, vad, emb)

    def isolate(x: np.ndarray, s: str) -> np.ndarray:
        ex.start(users[s].condition)
        return np.concatenate([ex.process(x), ex.flush()])

    # ---------------------------------------------------------------- verification
    log("scoring verification windows...")
    rep: dict = {"testset": man["source"], "speakers": len(ids), "config": cfg.to_dict()}
    for mode in (("raw", "extracted") if "verification" in parts else ()):
        gen, gen_t, imp, imp_t = [], [], [], []
        for s in ids:
            for x in tests[s]:
                w = windows(isolate(x, s) if mode == "extracted" else x, profiles)
                gen += w
                gen_t += [s] * len(w)
            for o in ids:
                if o != s:
                    w = windows(isolate(tests[o][0], s) if mode == "extracted" else tests[o][0], profiles)
                    imp += w
                    imp_t += [s] * len(w)
        rep[f"verification_{mode}"] = _far_frr(gen, imp, gen_t, imp_t, cfg.gate.open_threshold, cfg.gate.closed_set)

    # ---------------------------------------------------------------- gate on mixtures
    log("streaming mixtures through the full pipeline...")
    seg_stats: dict[str, list[float]] = {k: [] for k in
                                         ("interferer_alone_leak_db", "interferer_alone_open_frac",
                                          "target_alone_keep_db", "overlap_si_sdr_in", "overlap_si_sdr_out",
                                          "overlap_si_sdr_gain_db")}
    alerts, per_speaker = 0, []
    for s in (spk if "gate" in parts else []):
        d = testset / s["id"]
        mix, tgt = load_audio(d / "mix.wav"), load_audio(d / "mix_target.wav")
        calls = []
        lock = SpeakerLock(cfg, s["id"], users[s["id"]].condition, profiles, ex, vad, emb,
                           on_unverified=calls.append, background_verify=False)
        y, metas = lock.process_file(mix, 4096)
        lock.close()
        alerts += len(calls)
        ia, ib = s["segments"]["interferer_alone"]
        ta, tb = s["segments"]["target_alone"]
        oa, ob = s["segments"]["overlap"]
        seg_stats["interferer_alone_leak_db"].append(_db(np.sum(y[ia:ib] ** 2), np.sum(mix[ia:ib] ** 2)))
        opened = np.concatenate([np.full(m.samples, m.verified) for m in metas])[: len(y)]
        seg_stats["interferer_alone_open_frac"].append(float(opened[ia:ib].mean()))
        seg_stats["target_alone_keep_db"].append(_db(np.sum(y[ta + SR:tb] ** 2), np.sum(tgt[ta + SR:tb] ** 2)))
        sin, sout = si_sdr(mix[oa:ob], tgt[oa:ob]), si_sdr(y[oa:ob], tgt[oa:ob])
        seg_stats["overlap_si_sdr_in"].append(sin)
        seg_stats["overlap_si_sdr_out"].append(sout)
        seg_stats["overlap_si_sdr_gain_db"].append(sout - sin)
        per_speaker.append({"id": s["id"], **{k: round(float(v[-1]), 2) for k, v in seg_stats.items()}})
    if "gate" in parts:
        rep["gate"] = {k: round(float(np.mean(v)), 3) for k, v in seg_stats.items()}
        rep["gate"]["on_unverified_calls"] = alerts
        rep["gate"]["per_speaker"] = per_speaker

    # ---------------------------------------------------------------- extraction in -5 dB babble
    if "babble" in parts:
        log("measuring extraction SI-SDRi in -5 dB babble noise...")
        gains = []
        for s in spk:
            a = tests[s["id"]][0][: 8 * SR]
            bab = np.concatenate([tests[o][0][: len(a)] for o in ids if o != s["id"]][:4])
            bab = np.tile(bab, int(np.ceil(len(a) / max(1, len(bab)))))[: len(a)]
            bab = bab * (float(np.sqrt(np.mean(a ** 2))) * 10 ** (5 / 20)
                         / (float(np.sqrt(np.mean(bab ** 2))) + 1e-12))     # SNR = -5 dB
            mix = a + bab
            y = isolate(mix, s["id"])[: len(a)]
            gains.append(si_sdr(y, a) - si_sdr(mix, a))
        rep["babble_si_sdr"] = {"snr_db": -5.0, "si_sdr_gain_db_mean": round(float(np.mean(gains)), 2),
                               "si_sdr_gain_db_min": round(float(np.min(gains)), 2),
                               "per_speaker": dict(zip(ids, [round(float(g), 2) for g in gains])),
                               "target_db": 12.5}
    if "latency" not in parts:
        return rep

    # ---------------------------------------------------------------- latency / RTF
    log("measuring latency and real-time factor...")
    first = spk[0]["id"]
    mix = load_audio(testset / first / "mix.wav")
    lock = SpeakerLock(cfg, first, users[first].condition, profiles, ex, vad, emb, background_verify=True)
    F = cfg.frame_samples
    t0, compute = time.perf_counter(), []
    for i in range(0, len(mix), F):
        compute.append(lock.process(mix[i:i + F])[1].compute_ms)
    rtf = (time.perf_counter() - t0) / (len(mix) / SR)
    lock.reset()
    runner = LiveRunner(lock, lambda y, m: None).start()
    n = int(live_seconds * SR)
    audio = np.tile(mix, int(np.ceil(n / len(mix))))[:n]
    start = time.perf_counter()
    for i in range(0, n, F):
        time.sleep(max(0.0, start + (i + F) / SR - time.perf_counter()))
        runner.push(audio[i:i + F])
    runner.stop()
    lock.close()
    e2e = np.array(runner.e2e_ms)
    rep["latency"] = {
        "frame_ms": cfg.frame_ms, "algorithmic_ms": round(lock.latency_samples / SR * 1000, 1),
        "compute_per_frame_ms_p50_p95": [_pct(compute, 50), _pct(compute, 95)],
        "rtf_frame_by_frame": round(rtf, 3),
        "live_e2e_ms_p50": _pct(e2e, 50), "live_e2e_ms_p95": _pct(e2e, 95),
        "live_seconds": live_seconds, "live_dropped_samples": runner.dropped,
        "budget_ms": 150, "meets_budget_p95": bool(_pct(e2e, 95) < 150),
    }
    return rep
