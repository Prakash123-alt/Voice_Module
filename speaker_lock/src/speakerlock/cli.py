"""speakerlock CLI: enroll | run --mic | run --file | benchmark | users."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, Config


def _cfg(path: Optional[str]) -> Config:
    return Config.from_yaml(path) if path else Config()


def _record(seconds: float, device=None) -> np.ndarray:
    import sounddevice as sd  # noqa: PLC0415
    print(f"Recording {seconds:.0f} s - speak naturally now...", flush=True)
    x = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    return x[:, 0]


def cmd_enroll(a: argparse.Namespace) -> int:
    from .audio_io import load_audio  # noqa: PLC0415
    from .embedder import build_embedder  # noqa: PLC0415
    from .enroll import UserStore, enroll  # noqa: PLC0415
    from .extractor import build_extractor  # noqa: PLC0415
    from .vad import build_vad  # noqa: PLC0415
    cfg = _cfg(a.config)
    clip = load_audio(a.file) if a.file else _record(a.seconds, a.device)
    e = enroll(a.user, clip, build_extractor(cfg.extractor, cfg.device), build_embedder(cfg.verifier, cfg.device),
               build_vad(cfg.vad, cfg.frame_samples, cfg.device), cfg.vad.threshold)
    path = UserStore(a.store or cfg.store_dir).save(e)
    for w in e.meta["warnings"]:
        print("warning:", w)
    print(f"enrolled {a.user!r}: {e.meta['speech_s']} s of speech -> {path}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .audio_io import load_audio, save_audio  # noqa: PLC0415
    from .pipeline import SpeakerLock  # noqa: PLC0415
    cfg = _cfg(a.config)
    alerts = []
    lock = SpeakerLock.create(cfg, a.user, a.store, on_unverified=lambda i: (alerts.append(i), print(
        f"\n[on_unverified] {i['failures']} failing windows, best match {i['best_user']} ({i['score']:.2f})")),
        background_verify=not a.file)
    meta_f = open(a.meta, "w", encoding="utf-8") if a.meta else None
    out: list[np.ndarray] = []

    def sink(y: np.ndarray, m) -> None:
        out.append(y)
        if meta_f:
            meta_f.write(json.dumps(m.__dict__) + "\n")
        if not a.quiet:
            print(f"\r{'VERIFIED' if m.verified else 'blocked '}  score={m.score if m.score is not None else '-':>6}"
                  f"  latency={m.latency_ms:6.1f} ms  vad={m.vad:.2f}   ", end="", flush=True)

    try:
        if a.file:
            x = load_audio(a.file)
            t0 = time.perf_counter()
            for i in range(0, len(x), cfg.frame_samples):
                sink(*lock.process(x[i:i + cfg.frame_samples]))
            sink(*lock.flush())
            took = time.perf_counter() - t0
            print(f"\nprocessed {len(x) / SAMPLE_RATE:.1f} s in {took:.1f} s (RTF {took / (len(x) / SAMPLE_RATE):.2f})")
        else:
            import sounddevice as sd  # noqa: PLC0415

            from .stream import LiveRunner  # noqa: PLC0415
            runner = LiveRunner(lock, sink).start()
            print("Listening (Ctrl+C to stop)...")
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=a.device,
                                blocksize=cfg.frame_samples, callback=lambda d, *_: runner.push(d[:, 0].copy())):
                try:
                    while a.seconds <= 0 or sum(len(o) for o in out) < a.seconds * SAMPLE_RATE:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            runner.stop()
            if runner.e2e_ms:
                lat = np.array(runner.e2e_ms)
                print(f"\nend-to-end latency p50={np.percentile(lat, 50):.0f} ms p95={np.percentile(lat, 95):.0f} ms")
    finally:
        lock.close()
        if meta_f:
            meta_f.close()
    if a.out:
        save_audio(a.out, np.concatenate(out) if out else np.zeros(0, np.float32))
        print(f"clean audio -> {a.out}")
    return 0


def cmd_users(a: argparse.Namespace) -> int:
    from .enroll import UserStore  # noqa: PLC0415
    store = UserStore(a.store or _cfg(a.config).store_dir)
    if a.delete:
        store.delete(a.delete)
        print(f"deleted {a.delete}")
    for u in store.users():
        print(u, json.dumps(store.load(u).meta))
    return 0


def cmd_benchmark(a: argparse.Namespace) -> int:
    from .bench import run_benchmark  # noqa: PLC0415
    report = run_benchmark(_cfg(a.config), Path(a.testset).expanduser(), speakers=a.speakers,
                           live_seconds=a.live_seconds, parts=tuple(a.parts.split(",")))
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="speakerlock", description="Speaker-locked audio middleware")
    ap.add_argument("--config", help="YAML config (defaults built in)")
    ap.add_argument("--store", help="user store directory (default from config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="enroll a user from a ~5 s clip")
    e.add_argument("--user", required=True)
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--mic", action="store_true")
    e.add_argument("--seconds", type=float, default=6.0)
    e.add_argument("--device", default=None)

    r = sub.add_parser("run", help="stream audio through the filter")
    r.add_argument("--user", required=True)
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--mic", action="store_true")
    r.add_argument("--out", help="write the clean 16 kHz audio to this WAV")
    r.add_argument("--meta", help="write per-chunk metadata as JSON lines")
    r.add_argument("--seconds", type=float, default=0, help="mic: stop after N seconds (0 = until Ctrl+C)")
    r.add_argument("--device", default=None)
    r.add_argument("--quiet", action="store_true")

    b = sub.add_parser("benchmark", help="latency, RTF, FAR/FRR and leakage on a test set")
    b.add_argument("--testset", default="~/.speakerlock/testset")
    b.add_argument("--speakers", type=int, default=8)
    b.add_argument("--live-seconds", type=float, default=20.0)
    b.add_argument("--parts", default="verification,gate,babble,latency", help="comma-separated subset to run")
    b.add_argument("--out")

    u = sub.add_parser("users", help="list (or --delete) enrolled users")
    u.add_argument("--delete")

    a = ap.parse_args(argv)
    if getattr(a, "device", None) is not None and str(a.device).isdigit():
        a.device = int(a.device)
    return {"enroll": cmd_enroll, "run": cmd_run, "users": cmd_users, "benchmark": cmd_benchmark}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
