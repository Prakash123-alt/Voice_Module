"""Attack harness runner — Phase 4.

Loads WAV files from a directory tree and runs each through the Firewall,
recording pass/block decisions and latency per scenario.

Directory layout expected under ``audio_dir``::

    attack_audio/
      owner/            owner voice clips (for false-block rate)
      tv_audio/
      other_speaker/
      loopback/
      hidden_audio/
      cloned_voice/
      ultrasound/

Each sub-directory maps to a ScenarioKind by name.  Any WAV file inside is
treated as one trial.  Missing directories are silently skipped (the
corresponding scenario reports 0 trials).

Usage::

    from tests.attack_harness.runner import run_harness, print_report
    results = run_harness(firewall, audio_dir="attack_audio/")
    print_report(results)
"""
from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from tests.attack_harness.scenarios import ALL_SCENARIOS, AttackScenario, ScenarioKind
from voice_firewall._pipeline import Firewall


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    blocked: bool
    latency_ms: float
    flags: list[str] = field(default_factory=list)
    liveness_score: Optional[float] = None
    speaker_score: Optional[float] = None


@dataclass
class ScenarioResult:
    scenario: AttackScenario
    trials: list[TrialResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def block_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.blocked) / len(self.trials)

    @property
    def pass_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if not t.blocked) / len(self.trials)

    def _latencies(self) -> list[float]:
        return [t.latency_ms for t in self.trials]

    def p50_ms(self) -> float:
        s = sorted(self._latencies())
        return s[len(s) // 2] if s else 0.0

    def p95_ms(self) -> float:
        s = sorted(self._latencies())
        idx = min(len(s) - 1, int(len(s) * 0.95))
        return s[idx] if s else 0.0


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV file as float32 mono at target_sr."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()

    dtype = np.int16 if sampwidth == 2 else np.int32
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sampwidth == 2:
        audio /= 32768.0
    else:
        audio /= 2147483648.0

    # Mix to mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample if needed (linear interpolation — good enough for testing)
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        src_t = np.arange(len(audio), dtype=np.float64)
        dst_t = np.arange(n_out, dtype=np.float64) * (sr / target_sr)
        audio = np.interp(dst_t, src_t, audio.astype(np.float64)).astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

def _run_scenario(
    fw: Firewall,
    scenario: AttackScenario,
    wav_files: list[Path],
) -> ScenarioResult:
    result = ScenarioResult(scenario=scenario)
    for wav in wav_files:
        try:
            audio = _load_wav(wav)
            t0 = time.perf_counter()
            tagged = fw.process(audio)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            blocked = tagged is None
            trial = TrialResult(
                blocked=blocked,
                latency_ms=latency_ms,
                flags=list(tagged.transcript_flags) if tagged else [],
                liveness_score=tagged.liveness_score if tagged else None,
                speaker_score=tagged.speaker_score if tagged else None,
            )
            result.trials.append(trial)
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            break
    return result


def run_harness(
    firewall: Firewall,
    audio_dir: str | Path | None = None,
    scenarios: list[AttackScenario] | None = None,
) -> list[ScenarioResult]:
    """Run all scenarios against the firewall.

    Parameters
    ----------
    firewall:  An initialised Firewall instance.
    audio_dir: Root directory of the attack audio tree (optional).
               When None or missing, scenarios report 0 trials.
    scenarios: Override the scenario list (defaults to ALL_SCENARIOS).
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS

    base = Path(audio_dir) if audio_dir else None
    results = []
    for scenario in scenarios:
        wav_files: list[Path] = []
        if base is not None:
            kind_dir = base / scenario.kind.value
            if kind_dir.is_dir():
                wav_files = sorted(kind_dir.glob("*.wav"))
        if not wav_files:
            results.append(ScenarioResult(
                scenario=scenario,
                error="no audio files — add WAV files to run this scenario",
            ))
            continue
        results.append(_run_scenario(firewall, scenario, wav_files))
    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: list[ScenarioResult]) -> None:
    """Print a human-readable attack harness report to stdout."""
    print("\n" + "=" * 60)
    print("  Voice Firewall — Attack Harness Report")
    print("=" * 60)
    print()

    for r in results:
        if r.error:
            status = "SKIP"
        elif r.n == 0:
            status = "SKIP"
        elif r.scenario.expect_blocked and r.block_rate == 1.0:
            status = " OK "
        elif not r.scenario.expect_blocked:
            status = "INFO"   # expected to pass — no failure
        else:
            status = "WARN"

        print(f"[{status}] {r.scenario.name}  ({r.scenario.kind.value})")
        print(f"       {r.scenario.description}")
        if r.error:
            print(f"       {r.error}")
        elif r.n > 0:
            print(f"       trials={r.n}  block={r.block_rate:.0%}  pass={r.pass_rate:.0%}"
                  f"  p50={r.p50_ms():.1f}ms  p95={r.p95_ms():.1f}ms")
        if r.scenario.notes:
            print(f"       NOTE: {r.scenario.notes}")
        print()

    # Summary metrics
    total = sum(r.n for r in results)
    if total > 0:
        # Stranger-pass rate = pass_rate averaged over scenarios that expect_blocked
        attack_results = [r for r in results if r.scenario.expect_blocked and r.n > 0]
        if attack_results:
            stranger_pass = sum(r.pass_rate for r in attack_results) / len(attack_results)
            print(f"  Stranger-pass rate (avg, expect_blocked scenarios): "
                  f"{stranger_pass:.1%}")

        # Owner-false-block rate = block_rate for 'owner' scenarios (not expect_blocked)
        owner_results = [r for r in results
                         if not r.scenario.expect_blocked and r.n > 0
                         and r.scenario.kind.value == "owner"]
        if owner_results:
            false_block = sum(r.block_rate for r in owner_results) / len(owner_results)
            print(f"  Owner-false-block rate:                            "
                  f"{false_block:.1%}")

        all_latencies = [lat for r in results for t in r.trials for lat in [t.latency_ms]]
        if all_latencies:
            s = sorted(all_latencies)
            p50 = s[len(s) // 2]
            p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
            print(f"  Added latency  p50={p50:.1f}ms  p95={p95:.1f}ms  (all scenarios)")
    print()
