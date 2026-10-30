"""Run the voice assistant.

  python main.py                 # full voice loop
  python main.py --devices       # list audio devices
  python main.py --text          # keyboard in -> LLM -> TTS out (no mic; handy for testing LLM/TTS)
  python main.py --check         # verify OpenRouter key + model slugs, then exit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from voice.aec import build_aec
from voice.assistant import VoiceAssistant
from voice.audio_io import SoundDeviceInput, SoundDeviceOutput
from voice.chunker import SentenceChunker, clean_for_speech
from voice.config import load_config
from voice.denoise import build_denoiser
from voice.logging_utils import Metrics, log, setup_logging
from voice.openrouter import OpenRouterClient
from voice.speaker import ProfileSpeakerVerifier, build_embedder
from voice.stt import build_stt
from voice.tts import build_tts
from voice.vad import build_vad

logger = logging.getLogger("main")


async def check(cfg) -> int:
    client = OpenRouterClient(cfg.openrouter)
    try:
        models = set(await client.list_models())
        wanted = [cfg.llm.model] + ([cfg.stt.openrouter_model] if cfg.stt.mode == "openrouter" else []) \
            + ([cfg.tts.openrouter_model] if cfg.tts.mode == "openrouter" else [])
        ok = True
        for m in wanted:
            found = m in models
            ok &= found
            print(f"  {'OK ' if found else 'MISSING'} {m}")
            if not found:
                hint = [x for x in models if m.split("/")[0] in x][:5]
                if hint:
                    print(f"      candidates: {', '.join(hint)}")
        reply = []
        async for d in client.stream_chat([{"role": "user", "content": "Say 'ready' and nothing else."}], model=cfg.llm.model, max_tokens=5):
            reply.append(d)
        print(f"  LLM reply: {''.join(reply).strip()!r}")
        return 0 if ok else 1
    finally:
        await client.aclose()


async def text_mode(cfg) -> None:
    client = OpenRouterClient(cfg.openrouter)
    tts = build_tts(cfg.tts, client, cfg.audio.sample_rate)
    out = SoundDeviceOutput(cfg.audio.sample_rate, cfg.audio.frame_samples, cfg.audio.output_device, cfg.audio.output_gain)
    out.start()
    history: list[dict] = []
    print("Text mode. Type a message, Enter to send, Ctrl+C to quit.")
    try:
        while True:
            try:
                user = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            if not user.strip():
                continue
            msgs = [{"role": "system", "content": cfg.llm.system_prompt}, *history[-2 * cfg.llm.history_turns:],
                    {"role": "user", "content": user}]
            chunker, parts = SentenceChunker(cfg.tts.min_chunk_chars), []

            async def say(s: str) -> None:
                if s := clean_for_speech(s):
                    async for chunk in tts.synthesize(s):
                        out.play(chunk)

            async for d in client.stream_chat(msgs, model=cfg.llm.model, max_tokens=cfg.llm.max_tokens,
                                              temperature=cfg.llm.temperature, **cfg.llm.extra_body):
                parts.append(d)
                print(d, end="", flush=True)
                for s in chunker.feed(d):
                    await say(s)
            for s in chunker.flush():
                await say(s)
            print()
            history += [{"role": "user", "content": user}, {"role": "assistant", "content": "".join(parts)}]
            await out.wait_until_done()
    finally:
        out.stop()
        await client.aclose()


def build_assistant(cfg) -> VoiceAssistant:
    sr, fs = cfg.audio.sample_rate, cfg.audio.frame_samples
    client = OpenRouterClient(cfg.openrouter)
    embedder = build_embedder(cfg.speaker.backend)
    verifier = ProfileSpeakerVerifier.from_file(embedder, cfg.speaker.profile_path, threshold=cfg.speaker.threshold,
                                                min_segment_ms=cfg.speaker.min_segment_ms, sample_rate=sr)
    return VoiceAssistant(
        cfg,
        audio_in=SoundDeviceInput(sr, fs, cfg.audio.input_device, cfg.audio.input_gain),
        audio_out=SoundDeviceOutput(sr, fs, cfg.audio.output_device, cfg.audio.output_gain),
        aec=build_aec(cfg.aec, sr, fs),
        denoiser=build_denoiser(cfg.denoise.backend, sr),
        vad=build_vad(sr),
        verifier=verifier,
        stt=build_stt(cfg.stt, client, sr),
        llm=client,
        tts=build_tts(cfg.tts, client, sr),
        metrics=Metrics(),
    )


async def run(cfg) -> None:
    assistant = build_assistant(cfg)
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, assistant.stop)
    try:
        await assistant.run()
    except KeyboardInterrupt:
        pass
    finally:
        await assistant.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description="Local voice assistant (OpenRouter-only cloud).")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--text", action="store_true", help="keyboard input instead of microphone")
    ap.add_argument("--check", action="store_true", help="verify OpenRouter key and model slugs")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(args.log_level or cfg.logging.level, cfg.logging.json)
    if args.devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    try:
        if args.check:
            return asyncio.run(check(cfg))
        asyncio.run(text_mode(cfg) if args.text else run(cfg))
    except KeyboardInterrupt:
        log(logger, logging.INFO, "bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
