"""Guard 1 — Entry guard.

Stub for Phase 1 Step 1.  Full implementation arrives in Phase 1 Step 2.

Responsibility: reject connections that did not come from a trusted local caller.
- Bind to 127.0.0.1 only.
- Require a per-session random token on every request.
- Reject any request that carries a browser Origin header
  (browsers always send Origin; local native callers do not by default).
- Block WebSocket upgrade requests from web pages.
"""
from __future__ import annotations

from voice_firewall._types import GuardResult, GuardVerdict, PipelineContext, RawSegment
from voice_firewall.guards._base import GuardBase


class EntryGuard(GuardBase):
    """Locks the firewall to localhost + authenticated callers only.

    Phase 1 Step 1: stub — always passes audio segments (audio path guard is
    separate from connection-level guard; see service/server.py for the HTTP layer).
    """

    @property
    def name(self) -> str:
        return "g1_entry"

    def _check(self, segment: RawSegment, ctx: PipelineContext) -> GuardResult:
        # Audio path: segments that reach here have already been accepted at the
        # connection level (service layer).  No further per-frame check needed.
        return GuardResult(guard=self.name, verdict=GuardVerdict.PASS,
                           reason="connection-level check passed")
