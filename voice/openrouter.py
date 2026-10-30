"""The single OpenRouter client. All cloud AI traffic goes through here.

- httpx.AsyncClient, base_url https://openrouter.ai/api/v1, key from OPENROUTER_API_KEY.
- `stream_chat` yields text deltas from SSE. Cancelling the consuming task closes the
  response immediately (httpx aclose on the stream context exit).
- Retries with exponential backoff + jitter on 408/429/5xx and transport errors, honouring
  Retry-After. Retries only happen before the first token is delivered, so a half-spoken
  answer is never silently restarted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, AsyncIterator, Optional

import httpx

from .config import OpenRouterConfig
from .interfaces import LLMClient
from .logging_utils import log

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"OpenRouter HTTP {status}: {message}")
        self.status = status


class OpenRouterClient(LLMClient):
    def __init__(self, cfg: OpenRouterConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not cfg.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env)")
        self.cfg = cfg
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        if cfg.site_url:
            headers["HTTP-Referer"] = cfg.site_url
        if cfg.app_name:
            headers["X-Title"] = cfg.app_name
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), headers=headers,
                                         timeout=httpx.Timeout(cfg.timeout_s, connect=cfg.connect_timeout_s),
                                         transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- retry helpers -------------------------------------------------
    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.cfg.backoff_max_s)
            except ValueError:
                pass
        return min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * (2 ** attempt)) * (0.5 + random.random())

    async def _error_body(self, resp: httpx.Response) -> str:
        try:
            body = await resp.aread()
            data = json.loads(body)
            return str(data.get("error", {}).get("message") or data)[:300]
        except Exception:
            return resp.reason_phrase or ""

    # ---- public API ----------------------------------------------------
    async def chat(self, messages: list[dict], model: str, **extra: Any) -> dict:
        """Non-streaming completion. Returns the full JSON response (used for audio modalities)."""
        payload = {"model": model, "messages": messages, **extra}
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt >= self.cfg.max_retries:
                    raise
                await self._sleep_retry(attempt, None, f"transport: {e.__class__.__name__}")
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.cfg.max_retries:
                await self._sleep_retry(attempt, resp.headers.get("retry-after"), f"http {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise OpenRouterError(resp.status_code, await self._error_body(resp))
            return resp.json()
        raise OpenRouterError(0, "retries exhausted")

    async def stream_chat(self, messages: list[dict], model: str = "", **extra: Any) -> AsyncIterator[str]:
        payload = {"model": model, "messages": messages, "stream": True, **extra}
        for attempt in range(self.cfg.max_retries + 1):
            delivered = False
            try:
                async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                    if resp.status_code in RETRY_STATUS and attempt < self.cfg.max_retries:
                        await self._sleep_retry(attempt, resp.headers.get("retry-after"), f"http {resp.status_code}")
                        continue
                    if resp.status_code >= 400:
                        raise OpenRouterError(resp.status_code, await self._error_body(resp))
                    async for delta in self._iter_sse(resp):
                        delivered = True
                        yield delta
                    return
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if delivered or attempt >= self.cfg.max_retries:
                    raise
                await self._sleep_retry(attempt, None, f"transport: {e.__class__.__name__}")

    async def _sleep_retry(self, attempt: int, retry_after: Optional[str], why: str) -> None:
        delay = self._backoff(attempt, retry_after)
        log(logger, logging.WARNING, "openrouter retry", attempt=attempt + 1, reason=why, sleep_s=round(delay, 2))
        await asyncio.sleep(delay)

    @staticmethod
    def _parse_event(data: str) -> Optional[list[str]]:
        """Return text deltas for one SSE event payload, or None for [DONE]."""
        if data == "[DONE]":
            return None
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return []
        err = obj.get("error")
        if err and not obj.get("choices"):
            code = err.get("code", 0) if isinstance(err, dict) else 0
            msg = err.get("message", "stream error") if isinstance(err, dict) else str(err)
            raise OpenRouterError(code if isinstance(code, int) else 0, msg)
        return [c for ch in obj.get("choices") or [] if (c := (ch.get("delta") or {}).get("content"))]

    @classmethod
    async def _iter_sse(cls, resp: httpx.Response) -> AsyncIterator[str]:
        data_lines: list[str] = []
        async for line in resp.aiter_lines():
            if line.startswith(":"):  # OpenRouter sends ": OPENROUTER PROCESSING" keepalives
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line == "" and data_lines:  # blank line terminates an event
                deltas = cls._parse_event("\n".join(data_lines))
                data_lines.clear()
                if deltas is None:
                    return
                for d in deltas:
                    yield d
        if data_lines:  # stream closed without trailing blank line
            for d in cls._parse_event("\n".join(data_lines)) or []:
                yield d

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
