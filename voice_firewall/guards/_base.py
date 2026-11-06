"""GuardBase: the common interface every firewall guard must implement.

Design contracts
----------------
1. ``check()`` must NEVER raise an exception to its caller.  Any unhandled
   exception inside a guard is caught by the base-class wrapper and converted
   to a DROP result — **fail-closed**.
2. ``check()`` must be non-blocking and cheap on the calling thread.  Heavy
   work (embedding, STT) belongs in ``asyncio.to_thread`` inside the guard.
3. If a required resource (model file, profile) is missing, ``check()`` must
   return DROP, not PASS.  Guards are not allowed to silently degrade.
4. ``reset()`` must be safe to call at any time (device reconnect, restart).
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from voice_firewall._logging import log
from voice_firewall._types import GuardResult, GuardVerdict, PipelineContext, RawSegment

logger = logging.getLogger(__name__)


class GuardBase(ABC):
    """Abstract base for all six firewall guards.

    Subclasses implement ``_check()``; the public ``check()`` method adds the
    fail-closed exception wrapper and latency recording.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier used in logs and guard_trace."""

    @abstractmethod
    def _check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        """Guard-specific logic.  Raise freely — the base wrapper catches all."""

    # ------------------------------------------------------------------
    def check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        """Public entry point.  Fail-closed: any exception → DROP."""
        t0 = time.perf_counter()
        try:
            result = self._check(segment, ctx)
        except Exception as exc:
            log(logger, logging.ERROR, "guard error — dropping segment (fail-closed)",
                guard=self.name, err=repr(exc))
            result = GuardResult(
                guard=self.name,
                verdict=GuardVerdict.DROP,
                reason=f"internal error: {type(exc).__name__}: {exc}",
            )
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.guard = self.name   # always stamp the guard name
        return result

    def reset(self) -> None:
        """Called on pipeline restart or device reconnect.  Override if needed."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
