"""Entry guard: connection-level security for the Voice Firewall local service.

Three rules (applied in order to every HTTP and WebSocket request):

1. **Origin check** — reject requests that carry a browser-style Origin header.
   Browsers always send Origin on cross-origin requests and on every WebSocket
   upgrade; native local callers (Python processes, CLI tools) do not.
   An explicit allow-list in config overrides this for specific origins.

2. **Token check** — require the per-session random token, supplied as:
   ``Authorization: Bearer <token>``  or  ``?token=<value>`` (query string).
   The full token is never logged; only a short prefix is shown at startup.

3. **Bind address** — the server must listen on 127.0.0.1 only.
   This is enforced at uvicorn startup in ``server.run()``, not here.

Design
------
- ``SessionToken`` is generated once per process start (or read from
  ``VFW_SERVICE_TOKEN``).  It is never written to disk or returned by any API.
- ``check_connection()`` is a pure function so it can be unit-tested without HTTP.
- ``EntryGuardMiddleware`` is a thin ASGI wrapper around ``check_connection()``.
"""
from __future__ import annotations

import logging
import secrets
import sys
from typing import Any
from urllib.parse import parse_qs

from voice_firewall._config import EntryConfig
from voice_firewall._logging import log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session token
# ---------------------------------------------------------------------------

class SessionToken:
    """Random per-session bearer token.  Generated once; never stored to disk.

    Also holds issued *session-scoped* tokens (returned by POST /api/session).
    A session-scoped token only authorises WebSocket paths (/ws, /stream) —
    the master token is still required for every API endpoint.
    """

    def __init__(self, explicit: str = "") -> None:
        if explicit:
            self._token = explicit
            source = "env"
        else:
            self._token = secrets.token_urlsafe(32)
            source = "auto-generated"
        self._session_tokens: set[str] = set()

        # Print to stderr at startup — intentional, this is a session credential
        # the integrating process or the user needs to read.
        print(
            f"[voice-firewall] session token: {self._token[:6]}***  ({source})",
            file=sys.stderr,
            flush=True,
        )
        log(logger, logging.INFO, "session token ready",
            source=source, prefix=self._token[:4] + "***")

    def issue_session(self) -> str:
        """Mint a WS-scoped token.  Bounded to 512 live tokens."""
        if len(self._session_tokens) > 512:
            self._session_tokens.clear()  # emergency bound; sessions re-auth
        tok = secrets.token_urlsafe(24)
        self._session_tokens.add(tok)
        return tok

    def revoke_session(self, tok: str) -> None:
        self._session_tokens.discard(tok)

    def matches(self, candidate: str, scope: str = "full") -> bool:
        """Constant-time comparison — prevents timing-based token inference.

        scope="full"  → master token only.
        scope="ws"    → master token OR any issued session token.
        """
        if not candidate:
            return False
        if secrets.compare_digest(self._token, candidate):
            return True
        if scope == "ws" and candidate in self._session_tokens:
            return True
        return False

    @property
    def full(self) -> str:
        """The complete token value.  Only expose to the integrating process."""
        return self._token


# ---------------------------------------------------------------------------
# Origin validation helpers
# ---------------------------------------------------------------------------

# Browsers always send one of these schemes in the Origin header.
_BROWSER_SCHEMES: tuple[str, ...] = (
    "http://",
    "https://",
    "chrome-extension://",
    "moz-extension://",
    "safari-extension://",
)


def is_browser_origin(origin: str) -> bool:
    """Return True if the Origin value looks like it was sent by a browser."""
    return bool(origin) and any(origin.startswith(s) for s in _BROWSER_SCHEMES)


def origin_is_allowed(origin: str, allowed_origins: list[str]) -> bool:
    return origin in allowed_origins


def _origin_reject(headers: dict[str, str], cfg: EntryConfig) -> str | None:
    """Return the rejected Origin value, or None if the origin is acceptable."""
    origin = headers.get("origin", "")
    if origin and not origin_is_allowed(origin, cfg.allowed_origins) \
            and is_browser_origin(origin):
        return origin
    return None


# ---------------------------------------------------------------------------
# Pure connection check (testable without HTTP)
# ---------------------------------------------------------------------------

def check_connection(
    headers: dict[str, str],
    query_string: str,
    token: SessionToken,
    cfg: EntryConfig,
    scope: str = "full",
) -> tuple[bool, int, str]:
    """Decide whether to allow a connection.

    Parameters
    ----------
    headers:       Lowercase-keyed header dict (string values).
    query_string:  Raw query string from the request URL.
    token:         The active session token.
    cfg:           Entry guard configuration.
    scope:         "ws" allows session-issued tokens; "full" master only.

    Returns
    -------
    ``(allowed, http_status_code, reason)``
    """
    # --- Rule 1: Origin check ------------------------------------------------
    bad_origin = _origin_reject(headers, cfg)
    if bad_origin:
        log(logger, logging.WARNING, "request rejected — browser Origin",
            origin=bad_origin)
        return False, 403, f"browser Origin not permitted: {bad_origin}"
    origin = headers.get("origin", "")

    # --- Rule 2: Token check -------------------------------------------------
    auth = headers.get("authorization", "")
    candidate = ""

    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()

    if not candidate:
        # Accept token from ?token=... query parameter (useful for WS URLs)
        params = parse_qs(query_string)
        candidates = params.get("token", [])
        if candidates:
            candidate = candidates[0]

    if not candidate:
        log(logger, logging.WARNING, "request rejected — missing token",
            origin=origin or "(none)")
        return False, 401, "missing token"

    if not token.matches(candidate, scope=scope):
        log(logger, logging.WARNING, "request rejected — invalid token",
            origin=origin or "(none)")
        return False, 401, "invalid token"

    return True, 200, "ok"


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------

# Public-path policy
# ------------------
# GET requests to non-API paths (the served UI, /assets/*, /worklets/*, SPA
# routes like /admin) are public — they carry no secrets, and the Origin check
# still applies so foreign web pages cannot fetch them.  Every API endpoint
# and every WebSocket still requires a token.
#
# Session-issued tokens (from POST /api/session) only authorise /ws and
# /stream — never API endpoints.
_WS_TOKEN_PATHS: frozenset[str] = frozenset({"/ws", "/stream"})
_PROTECTED_GET_PREFIXES: tuple[str, ...] = ("/api", "/health", "/metrics",
                                          "/authorize", "/stream")


def _is_public_http(method: str, path: str) -> bool:
    """True for UI/static GET loads; False for APIs, WS upgrades, mutations."""
    if method != "GET":
        return False
    return not path.startswith(_PROTECTED_GET_PREFIXES)


class EntryGuardMiddleware:
    """ASGI middleware that enforces the entry guard on every HTTP/WebSocket request.

    Usage (FastAPI / Starlette)::

        app.add_middleware_raw(EntryGuardMiddleware, token=session_token, cfg=entry_cfg)

    Or via manual ASGI wrapping::

        app = EntryGuardMiddleware(app, token=session_token, cfg=entry_cfg)
    """

    def __init__(self, app: Any, token: SessionToken, cfg: EntryConfig) -> None:
        self._app = app
        self._token = token
        self._cfg = cfg

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        scope_type = scope.get("type")

        # Pass through lifespan and other non-HTTP scopes unchanged.
        if scope_type not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        # Build a lowercase string header dict.
        raw_headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in raw_headers
        }
        query_string = scope.get("query_string", b"").decode("latin-1")

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # UI/static GET loads are public (no secrets in them); Origin check
        # still applies so foreign web pages cannot fetch them.
        if scope_type == "http" and _is_public_http(method, path):
            bad_origin = _origin_reject(headers, self._cfg)
            if bad_origin:
                await self._reject_http(
                    send, 403, f"browser Origin not permitted: {bad_origin}"
                )
                return
            await self._app(scope, receive, send)
            return

        tok_scope = "ws" if path in _WS_TOKEN_PATHS else "full"
        allowed, status, reason = check_connection(
            headers, query_string, self._token, self._cfg, scope=tok_scope
        )

        if allowed:
            await self._app(scope, receive, send)
            return

        # --- Reject ---
        if scope_type == "http":
            await self._reject_http(send, status, reason)
        else:
            # WebSocket: consume the connect event, then close with 1008.
            # RFC 6455 §7.4.1: 1008 = Policy Violation.
            event = await receive()
            if event.get("type") == "websocket.connect":
                # Reason must be <= 123 bytes per RFC 6455.
                await send({
                    "type": "websocket.close",
                    "code": 1008,
                    "reason": reason[:123],
                })

    @staticmethod
    async def _reject_http(send: Any, status: int, reason: str) -> None:
        """Send a plain-text rejection response."""
        body = reason.encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })
