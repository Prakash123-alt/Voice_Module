"""Attack harness report generator.

Runs all defined scenarios and reports:
- stranger-pass rate
- owner-false-block rate
- added latency p50 / p95

Phase 1: skeleton only. Full replay implemented in Phase 3+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tests.attack_harness.scenarios import ALL_SCENARIOS, AttackScenario


@dataclass
class ScenarioResult:
    scenario: AttackScenario
    trials: int = 0
    blocked: int = 0
    passed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def block_rate(self) -> float:
        return self.blocked / self.trials if self.trials else 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.trials if self.trials else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[len(s) // 2]


def run_harness() -> list[ScenarioResult]:
    """Placeholder — returns empty results until Phase 3 wires in audio replay."""
    return [ScenarioResult(scenario=s, error="not implemented (Phase 3+)")
            for s in ALL_SCENARIOS]


def print_report(results: list[ScenarioResult]) -> None:
    print("\n=== Voice Firewall Attack Harness Report ===\n")
    for r in results:
        status = "SKIP" if r.error else ("OK" if r.block_rate == 1.0 else "WARN")
        print(f"[{status}] {r.scenario.name} ({r.scenario.kind.value})")
        print(f"       {r.scenario.description}")
        if r.error:
            print(f"       ERROR: {r.error}")
        else:
            print(f"       block={r.block_rate:.0%}  pass={r.pass_rate:.0%}  "
                  f"p50={r.p50_ms:.1f}ms  trials={r.trials}")
        if r.scenario.notes:
            print(f"       NOTE: {r.scenario.notes}")
        print()


if __name__ == "__main__":
    print_report(run_harness())
