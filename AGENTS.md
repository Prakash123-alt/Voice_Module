# AGENTS.md

## Commands
- Tests: `python -m pytest -q` (no audio hardware, models or network needed; ~2 s)
- Verify OpenRouter key + model slugs: `python main.py --check` (makes one tiny paid LLM call)
- Run: `python main.py` (needs `profile/voice.npy` from `python enroll.py` and `models/*.onnx` Piper voice)
- List audio devices: `python main.py --devices`
- Web UI: `python -m ui.server` (http://127.0.0.1:8765). Frontend is vanilla JS in `ui/static/`, no build;
  syntax-check with `node --check ui/static/app.js`.

## Environment notes
- Dev machine: Windows, Python 3.13, CPU only. `speexdsp` does not build on Windows; the default AEC is
  WebRTC AEC3 via `livekit` (local APM only).
- `resemblyzer` needs `webrtcvad-wheels` (the plain `webrtcvad` module imports removed `pkg_resources`).
- The Windows default input device on this machine is a virtual webcam mic (Iriun/DroidCam); set
  `audio.input_device` explicitly.
- Model slugs on OpenRouter change: `xiaomi/mimo-v2-flash` and `openai/gpt-4o-audio-preview` no longer exist;
  check with `--check` or `OpenRouterClient.list_models()`.

## SpeakerLock (`speaker_lock/` subpackage)
- Optional voice-isolation middleware: mic -> VoxLock extraction -> rolling d-vector gate ->
  only the enrolled user's voice reaches VAD/STT. Enable via `isolation.enabled` in config.yaml.
- Enroll separately from `profile/voice.npy`: `python -m speakerlock.cli enroll --user me --file <clip>`
  (run inside `speaker_lock/`; store at `~/.speakerlock/users`).
- `voice/isolation.py` wraps it in a worker thread (`LiveRunner`); `process_frame` only
  enqueues/dequeues - never run the extractor on the DSP thread (~30 ms per 32 ms frame here).
- Package tests: `cd speaker_lock && python -m pytest -q` (~2 min; needs downloaded ONNX models).
- Benchmark: `python -m speakerlock.cli benchmark` (test set at `~/.speakerlock/testset`).

## Conventions
- All cloud AI traffic goes only through `voice/openrouter.py` (`OpenRouterClient`).
- Each pipeline stage implements an ABC in `voice/interfaces.py`; audio is float32 16 kHz mono, 20 ms frames.
- Never do heavy work (embeddings, STT) on the DSP thread (`VoiceAssistant.process_frame`).
- Wrap async generators consumed in cancellable tasks with `contextlib.aclosing`.
- Log via `voice.logging_utils.log(logger, level, msg, **kv)`; never log raw audio or secrets.
- UI events come from `VoiceAssistant._emit(kind, **data)` -> `on_event` -> `ui.events.EventBus`; `on_event`
  can fire on the DSP thread, so the bus uses `call_soon_threadsafe`.
- `/api/config` must never return `openrouter.api_key` (tested in tests/test_ui_server.py).
