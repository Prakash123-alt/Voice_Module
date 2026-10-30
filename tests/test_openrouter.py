import asyncio
import json
from contextlib import aclosing

import httpx
import pytest

from voice.config import OpenRouterConfig
from voice.openrouter import OpenRouterClient, OpenRouterError


def cfg(**kw):
    base = dict(api_key="sk-or-v1-test", max_retries=3, backoff_base_s=0.0, backoff_max_s=0.0)
    base.update(kw)
    return OpenRouterConfig(**base)


def sse(*deltas, done=True) -> bytes:
    lines = [": OPENROUTER PROCESSING\n\n"]
    for d in deltas:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": d}}]}) + "\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


async def collect(client, **kw):
    return [d async for d in client.stream_chat([{"role": "user", "content": "hi"}], model="xiaomi/mimo-v2-flash", **kw)]


async def test_requires_api_key():
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(OpenRouterConfig(api_key=""))


async def test_stream_request_shape_and_deltas():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=sse("Hel", "lo", "!"), headers={"content-type": "text/event-stream"})

    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(handler))
    assert await collect(c, max_tokens=10) == ["Hel", "lo", "!"]
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-or-v1-test"
    assert seen["body"]["stream"] is True and seen["body"]["model"] == "xiaomi/mimo-v2-flash"
    assert seen["body"]["max_tokens"] == 10
    await c.aclose()


async def test_multiline_sse_event_and_no_done():
    body = b'data: {"choices": [{"delta":\ndata: {"content": "ok"}}]}\n\n'
    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    assert await collect(c) == ["ok"]


async def test_429_retried_then_success():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}}, headers={"retry-after": "0"})
        return httpx.Response(200, content=sse("done"))

    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(handler))
    assert await collect(c) == ["done"]
    assert len(calls) == 3


async def test_retries_exhausted_raises_with_status():
    c = OpenRouterClient(cfg(max_retries=2), transport=httpx.MockTransport(
        lambda r: httpx.Response(503, json={"error": {"message": "overloaded"}})))
    with pytest.raises(OpenRouterError) as ei:
        await collect(c)
    assert ei.value.status == 503 and "overloaded" in str(ei.value)


async def test_non_retryable_4xx_fails_fast():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OpenRouterError, match="bad key"):
        await collect(c)
    assert len(calls) == 1


async def test_transport_error_retried():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, content=sse("x"))

    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(handler))
    assert await collect(c) == ["x"]


async def test_mid_stream_error_event_raises():
    body = sse("partial", done=False) + b'data: {"error": {"code": 502, "message": "provider died"}}\n\n'
    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    with pytest.raises(OpenRouterError, match="provider died"):
        await collect(c)


async def test_chat_non_streaming_retry():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(handler))
    resp = await c.chat([{"role": "user", "content": "x"}], "m")
    assert resp["choices"][0]["message"]["content"] == "hi" and len(calls) == 2


def test_backoff_honours_retry_after_and_caps():
    c = OpenRouterClient(cfg(backoff_base_s=0.5, backoff_max_s=8.0))
    assert c._backoff(0, "2") == 2.0
    assert c._backoff(0, "100") == 8.0
    assert 0.25 <= c._backoff(0, None) <= 0.75
    assert c._backoff(10, None) <= 12.0


class HangingStream(httpx.AsyncByteStream):
    """Sends one delta then blocks forever, like a slow model. Records whether it was closed."""

    def __init__(self):
        self.closed = asyncio.Event()

    async def __aiter__(self):
        yield sse("first", done=False)
        await asyncio.sleep(3600)

    async def aclose(self):
        self.closed.set()


async def test_cancel_closes_stream_immediately():
    stream = HangingStream()
    c = OpenRouterClient(cfg(), transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)))
    got = asyncio.Event()

    async def consume():
        async with aclosing(c.stream_chat([{"role": "user", "content": "hi"}], model="m")) as s:
            async for _ in s:
                got.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(got.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(stream.closed.wait(), 0.1)  # HTTP response closed within 100 ms
