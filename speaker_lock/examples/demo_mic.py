"""Live mic demo: enroll once, then the agent only transcribes the enrolled speaker.

    python examples/demo_mic.py --user me --enroll-mic      # first time: record a ~6 s enrollment
    python examples/demo_mic.py --user me                   # afterwards: listen and transcribe

Have a second person talk, or play speech from a phone/TV - their audio is extracted away and,
if anything leaks through, the verification gate silences it before it reaches Whisper.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from speakerlock import Config  # noqa: E402
from speakerlock.config import SAMPLE_RATE  # noqa: E402
from speakerlock.pipeline import SpeakerLock  # noqa: E402
from speakerlock.stream import LiveRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="me")
    ap.add_argument("--config")
    ap.add_argument("--store")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--enroll-mic", action="store_true", help="record a fresh enrollment clip first")
    ap.add_argument("--seconds", type=float, default=0, help="listen for N seconds (0 = until Ctrl+C)")
    a = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice not installed: pip install sounddevice")

    cfg = Config.from_yaml(a.config) if a.config else Config()

    if a.enroll_mic:
        from speakerlock.cli import _record, cmd_enroll
        clip = _record(6.0, a.device)
        tmp = Path("demo_enroll.wav")
        from speakerlock.audio_io import save_audio
        save_audio(tmp, clip)
        rc = cmd_enroll(argparse.Namespace(config=a.config, user=a.user, file=str(tmp),
                                           mic=False, seconds=6.0, device=a.device, store=a.store))
        tmp.unlink(missing_ok=True)
        if rc != 0:
            return rc

    stt = None
    try:
        from speakerlock.integrations.whisper_adapter import WhisperAdapter, faster_whisper_transcriber
        stt = WhisperAdapter(faster_whisper_transcriber("base.en"))
    except ImportError:
        print("faster-whisper not installed - verified audio will be shown as gate events only")

    def on_unverified(info):
        print(f"\n  >> unverified speaker blocked (best match {info['best_user']!r}, "
              f"score {info['score']:.2f})")

    lock = SpeakerLock.create(cfg, a.user, a.store, on_unverified=on_unverified)

    def sink(y, m):
        if stt is not None:
            for text in stt.feed(y, m):
                print(f"\n  heard: {text!r}")
        else:
            print(f"\r  {'VERIFIED' if m.verified else 'blocked '}  "
                  f"score={m.score if m.score is not None else '-'}  vad={m.vad:.2f}   ",
                  end="", flush=True)

    runner = LiveRunner(lock, sink).start()
    t0 = time.perf_counter()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=a.device,
                            blocksize=cfg.frame_samples,
                            callback=lambda d, *_: runner.push(d[:, 0].copy())):
            print("listening (Ctrl+C to stop)...")
            while a.seconds <= 0 or time.perf_counter() - t0 < a.seconds:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    runner.stop()
    lock.close()
    if stt is not None:
        for text in stt.flush():
            print(f"  heard: {text!r}")
    if runner.dropped:
        print(f"note: dropped {runner.dropped / SAMPLE_RATE:.1f} s of audio (CPU could not keep up)")
    if runner.e2e_ms:
        lat = np.array(runner.e2e_ms)
        print(f"end-to-end latency p50={np.percentile(lat, 50):.0f} ms "
              f"p95={np.percentile(lat, 95):.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
