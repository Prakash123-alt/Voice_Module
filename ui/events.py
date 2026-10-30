"""Event bus between the assistant (any thread) and WebSocket clients (asyncio loop)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

from voice.logging_utils import redact

# Event types replayed to a newly connected browser so it can render recent context.
REPLAY_TYPES = {"transcript", "speaker", "turn", "bargein", "log", "error", "interrupted_reply"}


class EventBus:
    def __init__(self, replay: int = 400, client_queue: int = 1000) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._replay: deque = deque(maxlen=replay)
        self._client_queue = client_queue
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def publish(self, type_: str, **data: Any) -> None:
        """Call from the event loop thread."""
        ev = {"type": type_, "ts": round(time.time(), 3), **data}
        if type_ in REPLAY_TYPES:
            self._replay.append(ev)
        for q in list(self._clients):
            if q.full():  # slow client: drop its oldest event rather than block the pipeline
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(ev)

    def publish_threadsafe(self, type_: str, **data: Any) -> None:
        """Call from any thread (DSP thread, PortAudio callback, logging)."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(lambda: self.publish(type_, **data))
        except RuntimeError:  # loop shutting down
            pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._client_queue)
        for ev in self._replay:
            q.put_nowait(ev)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def clear_replay(self, types: set[str]) -> None:
        kept = [e for e in self._replay if e["type"] not in types]
        self._replay.clear()
        self._replay.extend(kept)


class BusLogHandler(logging.Handler):
    """Forwards log records (with their structured `kv`) to the bus, secrets redacted."""

    def __init__(self, bus: EventBus, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            kv = {k: redact(str(v)) for k, v in (getattr(record, "kv", None) or {}).items()}
            self.bus.publish_threadsafe("log", level=record.levelname, logger=record.name,
                                        msg=redact(record.getMessage()), kv=kv)
        except Exception:
            self.handleError(record)
