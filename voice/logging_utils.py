"""Structured logging + per-stage latency metrics. Never logs secrets or raw audio."""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

_SECRET_RE = re.compile(r"(sk-or-[A-Za-z0-9]*-?[A-Za-z0-9]{4})[A-Za-z0-9_\-]+|(Bearer\s+)\S+")


def redact(text: str) -> str:
    """Mask OpenRouter keys / bearer tokens anywhere in a log line (msg, kv fields, tracebacks)."""
    if "sk-or-" not in text and "Bearer" not in text:
        return text
    return _SECRET_RE.sub(lambda m: (m.group(1) or m.group(2)) + "***", text)


class _KVFormatter(logging.Formatter):
    def __init__(self, as_json: bool) -> None:
        super().__init__()
        self.as_json = as_json

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._format(record))

    def _format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "kv", None) or {}
        base = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
                "msg": record.getMessage(), **extra}
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        if self.as_json:
            return json.dumps(base, default=str)
        kv = " ".join(f"{k}={v}" for k, v in extra.items())
        line = f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {record.levelname:<7} {record.name}: {record.getMessage()} {kv}".rstrip()
        return line + (f"\n{base['exc']}" if "exc" in base else "")


def setup_logging(level: str = "INFO", as_json: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(_KVFormatter(as_json))
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("httpx", "httpcore", "numba", "urllib3", "faster_whisper", "livekit", "piper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **kv: Any) -> None:
    logger.log(level, msg, extra={"kv": kv})


class Metrics:
    """Per-stage latency accumulator. Call `timer("stage")` around work; `summary()` to print."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._marks: dict[str, float] = {}

    @contextmanager
    def timer(self, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000)

    def record(self, stage: str, ms: float) -> None:
        self._samples[stage].append(ms)

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def since(self, name: str) -> float | None:
        t = self._marks.get(name)
        return None if t is None else (time.perf_counter() - t) * 1000

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self._samples.items():
            s = sorted(v)
            out[k] = {"n": len(s), "p50_ms": round(s[len(s) // 2], 1),
                      "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1), "last_ms": round(v[-1], 1)}
        return out
