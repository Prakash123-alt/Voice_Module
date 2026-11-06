"""Attack scenario definitions for the firewall test harness.

Each scenario describes an attack type. The harness in report.py replays
them and measures: stranger-pass rate, owner-false-block rate, added latency.

Phase 1: scenario definitions only. Actual audio replay implemented in Phase 3+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ScenarioKind(str, Enum):
    TV_AUDIO        = "tv_audio"          # TV / YouTube / podcast in the room
    OTHER_SPEAKER   = "other_speaker"     # family member / friend voice
    LOOPBACK        = "loopback"          # command played from PC speakers
    HIDDEN_AUDIO    = "hidden_audio"      # instruction mixed quietly under owner speech
    CLONED_VOICE    = "cloned_voice"      # AI-cloned owner voice
    ULTRASOUND      = "ultrasound"        # near-ultrasound command


@dataclass
class AttackScenario:
    kind: ScenarioKind
    name: str
    description: str
    # Path to the audio clip used for replay (populated when available)
    audio_path: Optional[Path] = None
    # Expected behaviour: True = should be blocked, False = should pass
    expect_blocked: bool = True
    # Notes for the report (e.g., "EXPECTED to pass speaker guard — action guard is the safety net")
    notes: str = ""


# Registry of all defined scenarios (audio files added when available)
ALL_SCENARIOS: list[AttackScenario] = [
    AttackScenario(
        kind=ScenarioKind.TV_AUDIO,
        name="tv_commands",
        description="Commands spoken on TV/YouTube played in the room.",
        expect_blocked=True,
        notes="Speaker guard should reject — unknown voice embedding.",
    ),
    AttackScenario(
        kind=ScenarioKind.OTHER_SPEAKER,
        name="family_member",
        description="A different person in the household speaks a command.",
        expect_blocked=True,
        notes="Speaker guard should reject. Report false-accept rate honestly.",
    ),
    AttackScenario(
        kind=ScenarioKind.LOOPBACK,
        name="loopback_command",
        description="A command played from the PC's own speakers (e.g., via a web page).",
        expect_blocked=True,
        notes="Self-sound guard (G2) tags as loopback-suspect; speaker guard rejects.",
    ),
    AttackScenario(
        kind=ScenarioKind.HIDDEN_AUDIO,
        name="hidden_injection",
        description="An adversarial instruction mixed at low volume behind owner speech.",
        expect_blocked=False,  # May pass speaker guard — action guard is the safety net
        notes="WARNING: mixed audio passes speaker guard if owner is also present. "
              "Words guard flags mismatch. Action guard requires confirmation for risky actions.",
    ),
    AttackScenario(
        kind=ScenarioKind.CLONED_VOICE,
        name="cloned_owner_voice",
        description="AI-synthesised voice that mimics the enrolled owner.",
        expect_blocked=False,  # EXPECTED to pass speaker guard — by design
        notes="EXPECTED to pass the speaker guard. This is a known, documented limit. "
              "The action guard (G6) is the safety net. Never claim the firewall stops clones.",
    ),
    AttackScenario(
        kind=ScenarioKind.ULTRASOUND,
        name="ultrasound_command",
        description="A command encoded near or above 18 kHz, inaudible to humans.",
        expect_blocked=True,
        notes="Phase 3: ultrasound monitor detects and flags. Speaker guard rejects "
              "if no matching voice embedding.",
    ),
]
