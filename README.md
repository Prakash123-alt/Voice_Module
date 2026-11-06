# Voice Firewall

An on-device voice firewall that sits between the microphone and any voice AI agent.
It decides **who** is speaking, **where** the sound came from, and **what** the speech
is allowed to make the AI do.

```
Microphone → [VOICE FIREWALL] → Voice AI agent
```

Every audio segment leaving the firewall carries tags:

```
speaker_score, source (mic | loopback-suspect),
liveness_score, transcript_flags, risk_level
```

---

## What it does

| Guard | What it stops |
|---|---|
| **Entry guard** | Web pages and remote callers connecting to the local voice server |
| **Self-sound guard** | The PC's own speakers — YouTube, calls, the assistant's own TTS voice |
| **Speaker guard** | Other people in the room, TV/ads/podcasts |
| **Live-voice guard** | Raises a warning on synthetic/cloned voice (never sole proof) |
| **Words guard** | Empty transcripts, garbled audio, injection-suspect phrasing |
| **Action guard** | High-risk AI actions: always requires owner click/key press |

## What it does NOT stop (honest limits)

- **A perfect voice clone.** Voice is a probabilistic signal, not a password. The speaker
  guard will raise the bar; the action guard provides the safety net. We document this
  clearly and do not claim otherwise.
- **A physical attacker at the keyboard.** This is an audio firewall, not a full system
  guard.
- **Every possible adversarial audio technique.** The words guard and ultrasound monitor
  cover known attack vectors; novel techniques may slip through. Report them.
- **Injected audio when the enrolled owner is also speaking.** The firewall can warn but
  cannot reliably separate mixed sources in real time.
- **Commands masked by in-band music or noise.** The spectral check (background-energy
  ratio outside the 300–3400 Hz speech band) catches out-of-band injection but is a
  coarse heuristic. A command embedded inside music playing in the speech frequency
  range may not be flagged. A dedicated source-separation or masked-speech model
  would be required to close this gap.

Risky actions **always** require the owner's click or key press, independent of how
confident the speaker guard is. This is the final safety net.

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[mic,stt,dev]"

# Enroll your voice (creates profile/owner.npy)
python scripts/enroll.py

# Calibrate thresholds (writes them to firewall.yaml)
python scripts/calibrate.py

# Run tests
python -m pytest -q
```

## Architecture

Six guards run in order. Each is independently testable and individually
switchable via config:

```
Mic ──► G1 Entry ──► G2 Self-sound ──► G3 Speaker ──► G4 Live-voice ──► G5 Words ──► G6 Action ──► AI
         (locked        (loopback         (owner         (spoof           (transcript    (risk +
         connection)     AEC)              gate)          warning)         sanity)        confirm)
```

## Configuration

All thresholds come from `scripts/calibrate.py`. Do not set them by hand.
See `firewall.yaml` for all options. The API key (if any integration needs one)
goes in `.env` only — never in `firewall.yaml`.

## Licence

Apache-2.0. See LICENSE and NOTICE for dependency attributions.
