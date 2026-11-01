"""Continuous closed-set speaker verification + hysteresis gate.

RollingVerifier keeps the most recent `window_s` of *speech* frames (VAD-positive, after extraction) and,
every `hop_ms` of new speech, scores that window against every enrolled user. Scoring can run on a
background thread so the audio path never waits for an embedding.

HysteresisGate turns the stream of scores into a per-frame open/closed decision:
  open   when target score >= open_threshold (and, closed-set, the target is the best-matching user)
  stays  open while scores >= close_threshold; an ambiguous failing score only closes it after hold_ms since
         the last pass, but if another enrolled user clearly matches better it closes immediately
  turns  a pause >= turn_gap_ms restarts the scoring window, so each talker turn is judged on its own speech
  fades  over fade_ms on every transition (no clicks / chopped word edges)
  fails  N consecutive failing windows -> forced silence + on_unverified callback (once per streak)
  resets to its initial state after max_age_s without any speech (a new talker must be verified afresh)
"""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, GateConfig, VerifierConfig
from .interfaces import Embedder, cosine
from .liveness import liveness_score


@dataclass
class Score:
    target: float                     # cosine similarity to the target user
    best_user: Optional[str]          # best-matching enrolled user
    scores: dict[str, float] = field(default_factory=dict)
    at_sample: int = 0                # stream position (output samples) the window ends at
    liveness: Optional[float] = None  # heuristic replay/playback estimate (1.0 = likely live voice)


class RollingVerifier:
    def __init__(self, embedder: Embedder, profiles: dict[str, np.ndarray], target: str,
                 cfg: VerifierConfig, background: bool = True) -> None:
        if target not in profiles:
            raise KeyError(f"target user {target!r} is not enrolled")
        self.embedder, self.profiles, self.target, self.cfg = embedder, profiles, target, cfg
        self._frames: deque[tuple[int, np.ndarray]] = deque()   # (end sample, frame)
        self._speech = 0                                         # samples of speech in the window
        self._since = 0                                          # speech samples since the last score
        self._last_talk = -(1 << 62)                             # stream position where someone last talked
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="speakerlock-verify") if background else None
        self._pending: Optional[Future] = None
        self._lock = threading.Lock()
        self.new_turn = False                                    # set by push() on the first frame of a turn

    def reset(self) -> None:
        self._frames.clear()
        self._speech = self._since = 0
        self._last_talk = -(1 << 62)

    def push(self, frame: np.ndarray, is_speech: bool, end_sample: int,
             talking: Optional[bool] = None) -> Optional[Score]:
        """Add one (extracted) frame. `is_speech`: target-channel VAD (frame enters the window);
        `talking`: raw-input VAD, used to find turn boundaries (defaults to is_speech).
        Returns a score when one becomes available (sync: immediately; async: later)."""
        talking = is_speech if talking is None else talking
        max_age = int(self.cfg.max_age_s * SAMPLE_RATE)
        while self._frames and end_sample - self._frames[0][0] > max_age:
            self._speech -= self._frames.popleft()[1].shape[0]
        self.new_turn = False
        if talking:
            gap = self.cfg.turn_gap_ms * SAMPLE_RATE // 1000
            if gap and end_sample - frame.shape[0] - self._last_talk >= gap:
                self._frames.clear()                 # new turn: never let an earlier talker's speech vouch for it
                self._speech = self._since = 0
                self.new_turn = True
            self._last_talk = end_sample
        if is_speech:
            self._frames.append((end_sample, frame))
            self._speech += frame.shape[0]
            self._since += frame.shape[0]
            window = int(self.cfg.window_s * SAMPLE_RATE)
            while self._speech - self._frames[0][1].shape[0] >= window:
                self._speech -= self._frames.popleft()[1].shape[0]
        result = self._collect()
        settled = self._speech >= self.cfg.min_speech_s * SAMPLE_RATE
        need = self.cfg.min_speech_s if settled else self.cfg.turn_min_speech_s
        hop = self.cfg.hop_ms if settled else self.cfg.turn_hop_ms
        if self._speech >= need * SAMPLE_RATE and self._since >= hop * 16 and self._pending is None:
            self._since = 0
            audio = np.concatenate([f for _, f in self._frames])
            if self._pool is None:
                return self._score(audio, end_sample)
            self._pending = self._pool.submit(self._score, audio, end_sample)
        return result

    @property
    def idle(self) -> bool:
        """No speech left in the window (everything aged out)."""
        return not self._frames

    def _collect(self) -> Optional[Score]:
        if self._pending is not None and self._pending.done():
            fut, self._pending = self._pending, None
            return fut.result()
        return None

    def _score(self, audio: np.ndarray, at: int) -> Score:
        emb = self.embedder.embed(audio)
        scores = {u: cosine(emb, p) for u, p in self.profiles.items()}
        best = max(scores, key=scores.__getitem__)
        return Score(scores[self.target], best, scores, at, liveness=liveness_score(audio))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


class HysteresisGate:
    def __init__(self, cfg: GateConfig, target: str,
                 on_unverified: Optional[Callable[[dict], None]] = None) -> None:
        self.cfg, self.target, self.on_unverified = cfg, target, on_unverified
        self._fade = max(1, int(cfg.fade_ms * SAMPLE_RATE / 1000))
        self.reset()

    def reset(self) -> None:
        self.open = self.cfg.start_open
        self.failures = 0
        self.last_score: Optional[Score] = None
        self._gain = 1.0 if self.open else 0.0
        self._last_pass = -(1 << 62)
        self._turn_start: Optional[int] = None
        self._close_at: Optional[int] = None
        self._alerted = False
        self._attenuate = False

    def passes(self, s: Score) -> bool:
        thr = self.cfg.close_threshold if self.open else self.cfg.open_threshold
        return s.target >= thr and (not self.cfg.closed_set or s.best_user == self.target)

    def new_turn(self, now: int) -> None:
        """A new talker turn began at stream position `now`. strict_turns: withhold audio until
        this turn is verified; otherwise it may pass for turn_grace_ms (at grace_gain) and must
        produce a passing score before then."""
        self._turn_start = now
        self._attenuate = self.open and self.cfg.grace_gain < 1.0
        if self.cfg.strict_turns:
            self.open, self._close_at = False, None

    def update(self, score: Optional[Score], now: int, talking: bool = False) -> None:
        """Advance the decision to stream position `now` (samples) given an optional new score.
        `talking` = someone is speaking on the raw input (used to expire stale verifications)."""
        if score is not None:
            self.last_score = score
            if self.passes(score):
                self.open, self.failures, self._close_at, self._alerted = True, 0, None, False
                self._last_pass = now
            else:
                self.failures += 1
                other = (self.cfg.closed_set and score.best_user != self.target and
                         score.scores.get(score.best_user, -1.0) >= score.target + self.cfg.other_user_margin)
                if other:                      # someone else (enrolled) is talking: no grace period
                    self.open, self._close_at = False, None
                elif self.open and self._close_at is None:
                    self._close_at = self._last_pass + int(self.cfg.hold_ms * SAMPLE_RATE / 1000)
                if self.failures >= self.cfg.max_failures:
                    self.open, self._close_at = False, None
                    if not self._alerted and self.on_unverified is not None:
                        self._alerted = True
                        self.on_unverified({"failures": self.failures, "score": score.target,
                                            "best_user": score.best_user, "at_sample": now})
        if self._close_at is not None and now >= self._close_at:
            self.open, self._close_at = False, None
        # expire stale verification while speech continues: a new turn gets turn_grace_ms to
        # confirm itself; inside a turn a passing score must recur within score_ttl_ms.
        no_pass_this_turn = self._turn_start is not None and self._last_pass < self._turn_start
        if talking and self.open:
            if no_pass_this_turn and now - self._turn_start >= self.cfg.turn_grace_ms * 16:
                self.open, self._close_at = False, None
            elif (not no_pass_this_turn and self._last_pass > -(1 << 60)
                    and now - self._last_pass >= self.cfg.score_ttl_ms * 16):
                self.open, self._close_at = False, None
        self._attenuate = self.open and self.cfg.grace_gain < 1.0 and no_pass_this_turn

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Scale `frame` by the gate, ramping linearly over fade_ms toward the current state."""
        target = (self.cfg.grace_gain if self._attenuate else 1.0) if self.open else 0.0
        if self.cfg.invert:
            target = 1.0 - target    # reverse polarity: silence the enrolled voice, pass the rest
        if self._gain == target:
            return frame if target == 1.0 else (frame * self._gain if target else np.zeros_like(frame))
        n = frame.shape[0]
        ramp = np.clip(self._gain + (target - self._gain) * np.minimum(np.arange(1, n + 1) / self._fade, 1.0),
                       0.0, 1.0)
        self._gain = float(ramp[-1])
        return (frame * ramp).astype(np.float32)
