"""Assistant state machine: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> (INTERRUPTED) -> LISTENING."""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable, Optional

from .logging_utils import log

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING},
    State.LISTENING: {State.PROCESSING, State.IDLE},
    State.PROCESSING: {State.SPEAKING, State.LISTENING, State.IDLE},
    State.SPEAKING: {State.INTERRUPTED, State.LISTENING, State.IDLE},
    State.INTERRUPTED: {State.LISTENING, State.IDLE},
}


class InvalidTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, on_change: Optional[Callable[[State, State], None]] = None) -> None:
        self._state = State.IDLE
        self._since = time.perf_counter()
        self._on_change = on_change
        self.history: list[tuple[State, State, float]] = []

    @property
    def state(self) -> State:
        return self._state

    def can(self, new: State) -> bool:
        return new in TRANSITIONS[self._state]

    def transition(self, new: State, reason: str = "") -> None:
        if not self.can(new):
            raise InvalidTransition(f"{self._state.value} -> {new.value} not allowed")
        old, now = self._state, time.perf_counter()
        dwell_ms = (now - self._since) * 1000
        self._state, self._since = new, now
        self.history.append((old, new, dwell_ms))
        log(logger, logging.INFO, "state", **{"from": old.value, "to": new.value, "dwell_ms": round(dwell_ms), "reason": reason})
        if self._on_change:
            self._on_change(old, new)

    def try_transition(self, new: State, reason: str = "") -> bool:
        if not self.can(new):
            return False
        self.transition(new, reason)
        return True

    def is_(self, *states: State) -> bool:
        return self._state in states

    @property
    def dwell_ms(self) -> float:
        return (time.perf_counter() - self._since) * 1000
