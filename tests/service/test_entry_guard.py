"""Tests for the Entry Guard — connection-level security.

Three test classes:
1. TestSessionToken   — pure-Python token generation and comparison
2. TestOriginChecks   — pure-Python origin validation helpers
3. TestCheckConnection— pure-Python check_connection() logic
4. TestHTTPMiddleware — HTTP requests through the ASGI middleware (needs httpx)
5. TestWebSocketGuard — WebSocket upgrade rejection (needs httpx)
"""
from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from service.entry_guard import (
    EntryGuardMiddleware,
    SessionToken,
    check_connection,
    is_browser_origin,
    origin_is_allowed,
)
from voice_firewall._config import EntryConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def token() -> SessionToken:
    return SessionToken()


@pytest.fixture
def cfg() -> EntryConfig:
    return EntryConfig()


def _make_app(token: SessionToken, cfg: EntryConfig) -> Starlette:
    """Minimal Starlette app wrapped by the entry guard."""

    async def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    app = Starlette(routes=[
        Route("/", homepage),
        WebSocketRoute("/ws", ws_endpoint),
    ])
    return EntryGuardMiddleware(app, token=token, cfg=cfg)


# ---------------------------------------------------------------------------
# 1. SessionToken
# ---------------------------------------------------------------------------

class TestSessionToken:
    def test_tokens_are_unique(self) -> None:
        t1 = SessionToken()
        t2 = SessionToken()
        assert t1.full != t2.full

    def test_matches_self(self, token) -> None:
        assert token.matches(token.full) is True

    def test_rejects_empty_string(self, token) -> None:
        assert token.matches("") is False

    def test_rejects_wrong_token(self, token) -> None:
        assert token.matches("definitely-wrong") is False

    def test_explicit_token_accepted(self) -> None:
        t = SessionToken(explicit="my-secret-token-1234")
        assert t.matches("my-secret-token-1234") is True
        assert t.matches("my-secret-token-9999") is False

    def test_full_property_returns_string(self, token) -> None:
        assert isinstance(token.full, str)
        assert len(token.full) > 20

    def test_prefix_printed_to_stderr(self, capsys) -> None:
        t = SessionToken()
        captured = capsys.readouterr()
        # Token prefix is in stderr; full token is NOT
        assert t.full not in captured.err
        assert "***" in captured.err


# ---------------------------------------------------------------------------
# 2. Origin validation helpers
# ---------------------------------------------------------------------------

class TestOriginChecks:
    @pytest.mark.parametrize("origin", [
        "http://evil.com",
        "https://evil.com",
        "https://localhost:3000",
        "http://127.0.0.1:8080",
        "chrome-extension://abcdef",
        "moz-extension://abcdef",
        "safari-extension://abcdef",
    ])
    def test_browser_origins_detected(self, origin) -> None:
        assert is_browser_origin(origin) is True

    @pytest.mark.parametrize("origin", [
        "",
        "null",
        "file://local",
        "myapp://custom",
    ])
    def test_non_browser_origins(self, origin) -> None:
        assert is_browser_origin(origin) is False

    def test_allowed_list_exact_match(self) -> None:
        assert origin_is_allowed("http://my-app.local", ["http://my-app.local"]) is True

    def test_allowed_list_no_match(self) -> None:
        assert origin_is_allowed("http://evil.com", ["http://my-app.local"]) is False

    def test_empty_allowed_list(self) -> None:
        assert origin_is_allowed("http://any.com", []) is False


# ---------------------------------------------------------------------------
# 3. Pure check_connection() logic
# ---------------------------------------------------------------------------

class TestCheckConnection:
    def test_valid_bearer_token_allowed(self, token, cfg) -> None:
        ok, status, _ = check_connection(
            {"authorization": f"Bearer {token.full}"}, "", token, cfg
        )
        assert ok is True
        assert status == 200

    def test_valid_query_token_allowed(self, token, cfg) -> None:
        ok, status, _ = check_connection(
            {}, f"token={token.full}", token, cfg
        )
        assert ok is True
        assert status == 200

    def test_missing_token_rejected(self, token, cfg) -> None:
        ok, status, reason = check_connection({}, "", token, cfg)
        assert ok is False
        assert status == 401
        assert "missing" in reason

    def test_wrong_token_rejected(self, token, cfg) -> None:
        ok, status, reason = check_connection(
            {"authorization": "Bearer wrong-token"}, "", token, cfg
        )
        assert ok is False
        assert status == 401
        assert "invalid" in reason

    def test_browser_origin_blocked(self, token, cfg) -> None:
        ok, status, reason = check_connection(
            {"origin": "https://evil.com",
             "authorization": f"Bearer {token.full}"},
            "", token, cfg,
        )
        assert ok is False
        assert status == 403
        assert "Origin" in reason

    def test_allowed_origin_passes(self, token) -> None:
        cfg = EntryConfig(allowed_origins=["https://my-trusted-app.local"])
        ok, status, _ = check_connection(
            {"origin": "https://my-trusted-app.local",
             "authorization": f"Bearer {token.full}"},
            "", token, cfg,
        )
        assert ok is True
        assert status == 200

    def test_no_origin_no_token_rejected(self, token, cfg) -> None:
        ok, status, _ = check_connection({}, "", token, cfg)
        assert ok is False
        assert status == 401

    def test_case_insensitive_bearer(self, token, cfg) -> None:
        ok, _, _ = check_connection(
            {"authorization": f"BEARER {token.full}"}, "", token, cfg
        )
        # "BEARER" prefix — check_connection uses .lower() so it should pass
        assert ok is True

    def test_query_token_takes_effect_when_no_auth_header(self, token, cfg) -> None:
        import urllib.parse
        qs = urllib.parse.urlencode({"token": token.full})
        ok, status, _ = check_connection({}, qs, token, cfg)
        assert ok is True


# ---------------------------------------------------------------------------
# 4. HTTP middleware via TestClient
# ---------------------------------------------------------------------------

class TestHTTPMiddleware:
    @pytest.fixture
    def client(self, token, cfg) -> TestClient:
        app = _make_app(token, cfg)
        return TestClient(app, raise_server_exceptions=False)

    def test_valid_token_allowed(self, client, token) -> None:
        resp = client.get("/", headers={"Authorization": f"Bearer {token.full}"})
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_missing_token_returns_401(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, client) -> None:
        resp = client.get("/health", headers={"Authorization": "Bearer wrong-token-xyz"})
        assert resp.status_code == 401

    def test_browser_origin_returns_403(self, client, token) -> None:
        resp = client.get(
            "/",
            headers={
                "Authorization": f"Bearer {token.full}",
                "Origin": "https://malicious-page.com",
            },
        )
        assert resp.status_code == 403

    def test_local_app_origin_in_allowlist_passes(self, token) -> None:
        cfg = EntryConfig(allowed_origins=["http://my-local-app"])
        client = TestClient(_make_app(token, cfg), raise_server_exceptions=False)
        resp = client.get(
            "/",
            headers={
                "Authorization": f"Bearer {token.full}",
                "Origin": "http://my-local-app",
            },
        )
        assert resp.status_code == 200

    def test_query_string_token_allowed(self, client, token) -> None:
        resp = client.get(f"/?token={token.full}")
        assert resp.status_code == 200

    def test_query_string_wrong_token_rejected(self, client) -> None:
        resp = client.get("/health?token=totally-wrong")
        assert resp.status_code == 401

    def test_response_body_is_plain_text_on_reject(self, client) -> None:
        resp = client.get("/health")
        assert "token" in resp.text.lower()

    # --- Static UI exemption ----------------------------------------------

    def test_root_public_without_token(self, client) -> None:
        """GET / (UI) must pass the guard without a token.
        (200 if a dist build exists, 404 if not — the guard lets it through
        either way; a missing token would be 401.)"""
        resp = client.get("/")
        assert resp.status_code in (200, 404)

    def test_root_public_still_blocks_browser_origin(self, client) -> None:
        """Origin check applies even to the public static page."""
        resp = client.get("/", headers={"Origin": "https://evil.com"})
        assert resp.status_code == 403

    def test_root_post_still_requires_token(self, client) -> None:
        """Only GET is exempt — POST to / still needs the token."""
        resp = client.post("/")
        assert resp.status_code == 401

    def test_static_prefix_public_without_token(self, client) -> None:
        """/static/* is exempt like the root page."""
        resp = client.get("/static/index.html")
        assert resp.status_code in (200, 404)   # passes the guard; may 404 downstream

    def test_protected_path_still_401_when_public_exists(self, client) -> None:
        """Exemption is path-scoped — /health remains protected."""
        assert client.get("/health").status_code == 401


# ---------------------------------------------------------------------------
# 5. WebSocket guard
# ---------------------------------------------------------------------------

class TestWebSocketGuard:
    @pytest.fixture
    def client(self, token, cfg) -> TestClient:
        app = _make_app(token, cfg)
        return TestClient(app, raise_server_exceptions=False)

    def test_valid_token_ws_accepted(self, client, token) -> None:
        with client.websocket_connect(f"/ws?token={token.full}") as ws:
            msg = ws.receive_text()
        assert msg == "hello"

    def test_missing_token_ws_rejected(self, client) -> None:
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/ws") as ws:
                ws.receive_text()

    def test_wrong_token_ws_rejected(self, client) -> None:
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/ws?token=wrong") as ws:
                ws.receive_text()

    def test_browser_origin_ws_rejected(self, client, token) -> None:
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect(
                f"/ws?token={token.full}",
                headers={"Origin": "https://evil.com"},
            ) as ws:
                ws.receive_text()
