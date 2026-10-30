"""Two-tier automatic interruption manager.

Runs on every 20 ms mic frame in EVERY state. Pure logic - no asyncio, no I/O -
so it is fully unit-testable. Audio arrives already echo-cancelled (browser
getUserMedia AEC); a residual-echo energy guard covers what the AEC missed.

TIER 1 (fast): sustained VAD speech above the adaptive noise floor AND above the
    expected residual echo -> `duck` (~120-200 ms after onset).
TIER 2 (verify): embeddings on a growing window at 300/500/800 ms, done by the
    caller (async). `verify(score)` -> match = `confirmed`, miss by 800 ms =
    `false_alarm` -> `restore`. Without a speaker profile any >=300 ms stretch
    confirms directly (speaker lock off).
Backchannel: a ducked segment that ends < `backchannel_ms` without a confirm is
    a "yeah/uh-huh" -> `restore`, never a hard stop.
Cooldowns: `startup_cooldown_ms` after playback starts, `stop_cooldown_ms` after
    a confirmed stop - prevents chattering at boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DB_EPS = 1e-9


def db(x: float) -> float:
    import math
    return 20.0 * math.log10(max(x, DB_EPS))


@dataclass
class InterruptConfig:
    vad_threshold: float = 0.5
    tier1_ms: int = 140            # speech needed to duck playback
    confirm_ms: int = 300          # speech needed to confirm w/o a profile
    verify_points_ms: tuple = (300, 500, 800)
    verify_max_ms: int = 800       # no match by here -> false alarm
    backchannel_ms: int = 600      # shorter than this while ducked -> restore
    echo_margin_db: float = 6.0    # mic may exceed expected echo by this much
    startup_cooldown_ms: int = 200
    stop_cooldown_ms: int = 300
    min_score: float = 0.45        # speaker sim floor that scales up with window
    max_score: float = 0.62


@dataclass
class InterruptManager:
    cfg: InterruptConfig = field(default_factory=InterruptConfig)
    has_profile: bool = True
    playing: bool = False          # is the assistant currently streaming audio?
    noise_floor: float = 0.003     # adaptive RMS floor
    _speech_ms: int = 0            # consecutive user-ish speech while playing
    _checked_ms: int = -1          # last verify point issued
    phase: str = "idle"            # idle | ducked | confirmed_pending
    _cooldown_ms: int = 0

    def playback_started(self) -> None:
        self.playing = True
        self._speech_ms, self._checked_ms, self.phase = 0, -1, "idle"
        self._cooldown_ms = self.cfg.startup_cooldown_ms

    def playback_stopped(self) -> None:
        self.playing = False
        self._speech_ms, self._checked_ms, self.phase = 0, -1, "idle"

    def confirmed_stopped(self) -> None:
        """Call after the hard stop is done: arms the stop cooldown."""
        self.playback_stopped()
        self._cooldown_ms = self.cfg.stop_cooldown_ms

    def frame_ms(self) -> int:
        return 20

    def update(self, prob: float, mic_rms: float, ref_rms: float, frame_ms: int = 20) -> Optional[str]:
        """One frame -> None | "duck" | "verify" | "restore" | "speech_started".

        `verify` means: caller should run a speaker embedding on the current
        window NOW (async) and report via verify_result()."""
        # adaptive noise floor: track quiet frames when not playing speech context
        if mic_rms < self.noise_floor * 4:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * mic_rms
        if self._cooldown_ms > 0:
            self._cooldown_ms = max(0, self._cooldown_ms - frame_ms)
        if not self.playing or self._cooldown_ms > 0 or self.phase == "confirmed_pending":
            return None
        if prob < self.cfg.vad_threshold or mic_rms < self.noise_floor * 3:
            if self._speech_ms and self.phase == "ducked":
                out = "restore" if self._speech_ms < self.cfg.backchannel_ms else None
                self._speech_ms, self.phase = 0, "idle"
                return out
            self._speech_ms = 0
            return None
        # residual-echo guard: energy explained by the played reference is echo
        if db(mic_rms) < db(ref_rms) + self.cfg.echo_margin_db and ref_rms > self.noise_floor:
            self._speech_ms = 0
            return None
        prev = self._speech_ms
        self._speech_ms += frame_ms
        if self.phase == "idle" and self._speech_ms >= self.cfg.tier1_ms:
            self.phase = "ducked"
            return "duck"
        if self.phase == "ducked":
            nxt = next((p for p in self.cfg.verify_points_ms if p > self._checked_ms), None)
            if not self.has_profile and self._speech_ms >= self.cfg.confirm_ms:
                return "verify"           # no profile: caller auto-confirms
            if nxt is not None and self._speech_ms >= nxt:
                self._checked_ms = nxt
                return "verify"
            if self._speech_ms >= self.cfg.verify_max_ms:
                self.phase = "idle"       # exhausted the window without a match
                self._speech_ms = 0
                return "restore"
        return None

    def score_threshold(self, window_ms: int) -> float:
        """Similarity bar scales with how much speech the verifier saw."""
        lo, hi = self.cfg.min_score, self.cfg.max_score
        pts = self.cfg.verify_points_ms
        if window_ms <= pts[0]:
            return lo
        if window_ms >= pts[-1]:
            return hi
        span = pts[-1] - pts[0]
        return lo + (hi - lo) * (window_ms - pts[0]) / span

    def verify_result(self, accepted: bool, window_ms: int) -> Optional[str]:
        """Report the async embedding result -> "confirmed" | None (keep waiting)."""
        if self.phase != "ducked":
            return None
        if accepted:
            self.phase = "confirmed_pending"
            return "confirmed"
        if window_ms >= self.cfg.verify_max_ms:
            self.phase, self._speech_ms = "idle", 0
            return "restore"
        return None
