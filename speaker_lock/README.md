# SpeakerLock

Speaker-locked audio middleware. It sits in front of a speech-to-text engine (faster-whisper,
Deepgram, Pipecat, or anything that accepts 16 kHz mono PCM) and outputs a clean stream that
contains **only the enrolled user's voice**: background chatter, crosstalk, TV audio, and other
unenrolled speakers are removed by target-speaker extraction, and whatever leaks through is
silenced by a rolling speaker-verification gate.

```
mic / WebRTC / file  (16 kHz mono, any chunk size)
   |
   +-> RingBuffer (re-blocks into 32 ms frames)
   +-> Extractor  (VoxLock ONNX, causal, locked to the enrolled voice)
   +-> VAD x2     (Silero ONNX: extracted channel -> what is target speech,
   |               raw channel    -> is anyone talking, for turn boundaries)
   +-> RollingVerifier (background thread: d-vector on a 1.6 s window, every 160-320 ms)
   +-> HysteresisGate (open/close thresholds, hold, fades, turn grace, score TTL)
   |
clean 16 kHz PCM + per-chunk metadata {verified, score, best_user, latency_ms, vad, liveness, ...}
```

## Install

```bash
pip install -e .            # core: numpy, onnxruntime, pyyaml, soundfile
pip install -e .[mic]       # + sounddevice for --mic and the live demo
pip install -e .[whisper]   # + faster-whisper for the STT adapter/demo
pip install -e .[deepgram,pipecat,export,dev]
```

All ONNX models download automatically on first use (Silero VAD, VoxLock, d-vector) and are
cached under `~/.speakerlock/` with SHA-256 verification.

## Quick start

```bash
# 1. enroll a user from a ~5 s clip (or --mic to record one)
speakerlock enroll --user me --file my_voice.wav

# 2. clean an audio file: only the enrolled speaker survives
speakerlock run --user me --file mixed_conversation.wav --out clean.wav --meta meta.jsonl

# 3. live microphone
speakerlock run --user me --mic

# 4. demo (no mic needed): scene = target + interferer + overlap + noise -> gate -> Whisper
python examples/demo_agent.py

# 5. benchmark on the LibriSpeech dev-clean test set
speakerlock benchmark                       # needs scripts/make_testset.py first
python scripts/make_testset.py              # downloads ~340 MB, builds mixtures
```

Multiple users can be enrolled (`speakerlock users` lists them); the closed-set gate requires
the *enrolled target* to be the best match, so a second enrolled speaker is rejected faster
than an unknown one.

## Python API

```python
from speakerlock import Config
from speakerlock.pipeline import SpeakerLock

def alert(info):                       # N consecutive unverified windows -> silence
    print("unverified speaker:", info)

lock = SpeakerLock.create(Config(), user_id="me", on_unverified=alert)
for chunk in mic_frames:               # any chunk size, float32 16 kHz mono
    pcm, meta = lock.process(chunk)    # -> clean pcm (same length, 1:1 with input)
    stt.feed(pcm)                      # meta.verified / meta.score / meta.latency_ms ...
pcm, meta = lock.flush()
lock.close()
```

For real-time audio (PortAudio/WebRTC callbacks) use `speakerlock.stream.LiveRunner`, which
decouples the callback from processing and bounds the backlog instead of drifting:

```python
from speakerlock.stream import LiveRunner
runner = LiveRunner(lock, sink=lambda pcm, meta: stt.feed(pcm)).start()
# in the audio callback: runner.push(frame)
```

## How the gate works

The extractor removes non-target speech sample-by-sample (≈24 ms algorithmic delay), but it
needs roughly a second to adapt to a *new* voice. The verifier makes the final pass/block
decision — and a reliable speaker score needs ~1.5 s of audio, so the score always trails the
signal. The gate closes that gap with turn-aware rules:

- **Turns** are segmented on the *raw* input VAD ("someone is talking"), so suppression cannot
  fragment a talker's turn. A pause of `turn_gap_ms` starts a new turn.
- **`strict_turns: true` (default)**: every new turn is silent until its first passing score
  (~0.4-0.6 s of speech). Nothing leaks at turn boundaries; the cost is that the start of the
  target's own turns is also withheld.
- **`strict_turns: false`**: a new turn may pass for `turn_grace_ms` while it is being verified
  (bounded leak), optionally attenuated by `grace_gain` instead of passed at full volume.
- **`score_ttl_ms`**: while someone is talking, a passing score must repeat at least this often —
  a stale verification cannot keep the gate open through a speaker change mid-turn.
- **Hysteresis**: opens at `open_threshold`, stays open down to `close_threshold`, with `hold_ms`
  and `fade_ms` so words are not chopped and there are no clicks.
- **`max_failures`** consecutive failing windows fire `on_unverified` (your hook to warn / log).
- **`gate.invert: true` = reverse polarity** — mute ONLY the enrolled voice and pass everything
  else (privacy-redaction mode: the target's speech becomes silence, bystanders pass).
- **`liveness` token**: each scored window also carries a heuristic replay/playback estimate
  (spectral flatness + >4 kHz energy + clipping; 1.0 = likely a live voice). It is metadata for
  your downstream policy, *not* a certified anti-spoof model — pair with an AASIST-class
  detector for adversarial deployments.

## Measured on this laptop (i5-class CPU, ONNX int8)

Benchmark (`speakerlock benchmark`, 8 LibriSpeech dev-clean speakers, real read speech mixed
with cross-talk and noise — see `benchmarks/results.json` for the full report):

| metric | value |
|---|---|
| end-to-end latency (live, p50 / p95) | 85 / 110 ms (target < 150 ms) |
| algorithmic delay | 56 ms (extractor 24 ms + frame re-blocking) |
| real-time factor (32 ms frames) | 0.91, 0 dropped samples in 20 s live run |
| interferer-alone leak, mean | -68.8 dB (open only 3.8 % of the segment) |
| interferer-alone leak, worst pair | -36.6 dB |
| target-alone keep | -0.70 dB (cost of the strict-turn start clip) |
| overlap SI-SDR gain, mean | -1.7 dB (see Limitations) |
| SI-SDRi in -5 dB babble, mean (min) | 6.58 dB (-2.38) - see Limitations |
| verification EER (raw / extracted) | 2.36 % / 2.36 % |
| FAR / FRR at threshold 0.72 (extracted) | 0 % / 5.2 % |
| extractor cost | ~29-32 ms per 32 ms frame on this laptop |

**Latency flags.** The VoxLock extractor is the bottleneck: on a weak CPU its real-time factor
is slightly above 1 for 32 ms chunks, so `LiveRunner` batches frames under load and latency
settles higher rather than drifting. On a faster CPU (or with `precision: fp32` on GPU) it
sits comfortably under 150 ms. The verifier never adds audio-path latency — it runs on a
background thread.

## Swap the models

Every stage is behind an interface (`src/speakerlock/interfaces.py`): `VAD`, `Extractor`,
`Embedder`. Register alternatives in `build_vad` / `build_extractor` / `build_embedder`, or pass
your own objects into `SpeakerLock(...)`. The shipped default for extraction is
[VoxLock](https://github.com/arbazsiddiqui/voxlock) (Apache-2.0, causal BSRNN, ONNX, int8), for
VAD Silero (MIT), and for verification a GE2E d-vector exported to ONNX (Resemblyzer weights,
re-exported via `scripts/export_onnx.py`). An ECAPA-TDNN embedder drops in the same way.

## Integrations

- **faster-whisper**: `speakerlock.integrations.whisper_adapter.WhisperAdapter` accumulates
  *only verified* speech into utterances and transcribes on endpointing. See `demo_agent.py`.
- **Deepgram streaming**: `speakerlock.integrations.deepgram_adapter.DeepgramAdapter` forwards
  clean PCM over the Deepgram WebSocket. Needs `DEEPGRAM_API_KEY`; the protocol is unit-tested
  with a mocked socket.
- **Pipecat**: `speakerlock.integrations.pipecat_filter.SpeakerLockFilter` is a
  `BaseAudioFilter` — drop it into a Pipecat pipeline in front of the STT node.
- **LiveKit**: `speakerlock.integrations.livekit_adapter.LiveKitSpeakerLock` filters
  `rtc.AudioFrame`s (duck-typed, mono int16) before the agent/STT — `frame = gate.process(frame)`.
- **Voice_Module**: see "Voice_Module integration" below.

## ONNX / edge deployment

- VoxLock and Silero already ship ONNX graphs; the d-vector is exported by
  `python scripts/export_onnx.py` (requires `.[export]`) with a torch<->ONNX parity check.
- The whole runtime path is ONNX Runtime only — no torch needed at the edge.
- Per-thread session control: `extractor.threads` / `verifier.threads`; on a 2-core edge device
  pin extractor to 1 thread and expect ~2-3x the laptop latency — the gate logic is trivial.
- `precision: int8-full` is the default extractor graph; fp32 is ~3x slower but slightly better
  isolation on hard speaker pairs.

## Testing

```bash
python -m pytest -q     # unit + integration tests (needs the downloaded models; ~2-3 min)
```

## Voice_Module integration

`Voice_Module`'s `VoiceAssistant` mounts SpeakerLock as an optional stage (`voice/isolation.py`).
Enable it in `Voice_Module/config.yaml`:

```yaml
isolation:
  enabled: true
  user_id: me                  # must exist in the SpeakerLock store
  store_dir: ~/.speakerlock/users
```

`process_frame` pushes each post-AEC/denoise mic frame into the stage and continues with the
*gated* frame (silence while the worker catches up, so the stream just lags ~100 ms). All heavy
work — extraction, embedding, verification — stays off the DSP thread on the `LiveRunner`
worker; VAD, the segmenter, speaker verification and barge-in downstream see only the enrolled
voice. Enrollment is separate from `profile/voice.npy` (Resemblyzer): run
`python -m speakerlock.cli enroll --user me --file <clip>` once. If the user is missing the
assistant logs a warning and runs without isolation.

## Layout

```
src/speakerlock/
    config.py       Config/VerifierConfig/GateConfig dataclasses + YAML
    interfaces.py   VAD / Extractor / Embedder / TargetCondition ABCs
    buffer.py       RingBuffer: arbitrary chunks -> fixed frames
    vad.py          Silero ONNX + energy fallback
    extractor.py    VoxLock streaming extraction (carried LSTM state)
    embedder.py     d-vector ONNX embedder
    verifier.py     RollingVerifier + HysteresisGate
    enroll.py       enrollment + multi-user store (npz + json)
    pipeline.py     SpeakerLock: process()/flush() -> (pcm, ChunkMeta)
    stream.py       LiveRunner: callback-safe real-time driving
    models.py       pinned, checksum-verified model downloads
    audio_io.py     WAV load/save helpers
    cli.py          enroll | run --mic | run --file | benchmark | users
    bench.py        latency/RTF/FAR/FRR/leakage report
    testset.py      LibriSpeech mixture builder
    export.py       d-vector ONNX export
    integrations/   whisper_adapter, deepgram_adapter, pipecat_filter
examples/           demo_agent.py (one command, no mic), demo_mic.py
scripts/            export_onnx.py, make_testset.py, make_test_assets.py
benchmarks/         results.json
tests/              unit + integration tests
```

## Limitations (honest ones)

- The first ~0.5-0.7 s of *any* new turn is unverified speech — strict turns silence it (target
  words clipped) while non-strict passes/attenuates it (interferer leaks). There is no free
  lunch without lookahead, which would blow the 150 ms budget.
- On acoustically similar speaker pairs the extractor sometimes fails to separate; during heavy
  overlap those pairs lose the target too (safe side: silence, not crosstalk).
- In dense −5 dB babble the SI-SDRi averages 6.6 dB but is speaker-dependent (range −2.4 to
  +18.8 dB) — below the ~12.5 dB enterprise target; that needs a stronger TSE model, not tuning.
- The verifier is a GE2E d-vector — good, not state of the art. Swap in ECAPA-TDNN behind the
  `Embedder` interface for better short-utterance accuracy.
