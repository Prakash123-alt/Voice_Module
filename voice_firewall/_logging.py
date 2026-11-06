"""Structured logging, secret redaction, and bounded ring-buffer metrics.

Adapted from VoiceModule voice/logging_utils.py (same author).
Changes: ring-buffer metrics (deque with maxlen), extended secret patterns,
per-frame latency budget warning.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# Patterns that must never appear in logs.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(sk-or-[A-Za-z0-9]*-?[A-Za-z0-9]{4})[A-Za-z0-9_\-]+"),  # OpenRouter key
    re.compile(r"(Bearer\s+)\S+"),                                           # Bearer token
    re.compile(r"(hf_)[A-Za-z0-9]{16,}"),                                   # Hugging Face token
    re.compile(r"(VFW_SERVICE_TOKEN\s*=\s*)\S+"),                            # service token in env lines
]


def redact(text: str) -> str:
    """Mask known secret patterns anywhere in a log line.

    Call sites never need to remember to redact — the formatter always runs this.
    """
    for rx in _SECRET_PATTERNS:
        text = rx.sub(lambda m: m.group(1) + "***", text)
    return text


# ---------------------------------------------------------------------------
# KV log formatter
# ---------------------------------------------------------------------------

class _KVFormatter(logging.Formatter):
    """Human-readable key=value or JSON structured log lines."""

    def __init__(self, as_json: bool = False) -> None:
        super().__init__()
        self.as_json = as_json

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._format(record))

    def _format(self, record: logging.LogRecord) -> str:
        extra: dict[str, Any] = getattr(record, "kv", None) or {}
        if self.as_json:
            base: dict[str, Any] = {
                "ts": round(record.created, 3),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                **extra,
            }
            if record.exc_info:
                base["exc"] = self.formatException(record.exc_info)
            return json.dumps(base, default=str)
        kv = " ".join(f"{k}={v}" for k, v in extra.items())
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        line = f"{ts} {record.levelname:<7} {record.name}: {record.getMessage()} {kv}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO", as_json: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(_KVFormatter(as_json))
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "numba", "urllib3", "faster_whisper",
                  "livekit", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **kv: Any) -> None:
    """Emit a structured log line. Keys/values are attached as `kv` extra."""
    logger.log(level, msg, extra={"kv": kv})


# ---------------------------------------------------------------------------
# Bounded ring-buffer metrics
# ---------------------------------------------------------------------------

@dataclass
class _Stage:
    """Single-stage latency ring buffer (bounded memory)."""
    name: str
    maxlen: int
    _samples: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.maxlen)

    def record(self, ms: float) -> None:
        self._samples.append(ms)

    def reset(self) -> None:
        self._samples.clear()

    def summary(self) -> dict[str, float | int]:
        s = sorted(self._samples)
        n = len(s)
        if n == 0:
            return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0,
                    "mean_ms": 0.0, "last_ms": 0.0}
        mean_ms = round(sum(self._samples) / n, 2)
        return {
            "n": n,
            "p50_ms": round(s[n // 2], 2),
            "p95_ms": round(s[min(n - 1, int(n * 0.95))], 2),
            "max_ms": round(s[-1], 2),
            "mean_ms": mean_ms,
            "last_ms": round(list(self._samples)[-1], 2),
        }


class Metrics:
    """Per-stage latency accumulator with fixed-size ring buffers.

    Memory is bounded: each stage keeps at most `ring_buffer_size` samples.
    Use `timer("stage")` as a context manager, or `record("stage", ms)` directly.
    """

    def __init__(self, ring_buffer_size: int = 500) -> None:
        self._maxlen = ring_buffer_size
        self._stages: dict[str, _Stage] = {}

    @property
    def max_samples(self) -> int:
        """Maximum number of samples kept per stage (ring-buffer bound)."""
        return self._maxlen

    @property
    def stages(self) -> list[str]:
        """Names of all stages that have recorded at least one sample."""
        return list(self._stages.keys())

    def _stage(self, name: str) -> _Stage:
        if name not in self._stages:
            self._stages[name] = _Stage(name, self._maxlen)
        return self._stages[name]

    @contextmanager
    def timer(self, stage: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stage(stage).record((time.perf_counter() - t0) * 1000)

    def record(self, stage: str, ms: float) -> None:
        self._stage(stage).record(ms)

    def reset(self) -> None:
        """Clear all ring buffers (call on pipeline restart)."""
        for s in self._stages.values():
            s.reset()

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return per-stage latency statistics.

        Keys per stage: ``n``, ``p50_ms``, ``p95_ms``, ``max_ms``,
        ``mean_ms``, ``last_ms``.
        """
        return {name: s.summary() for name, s in self._stages.items()}

    def latency_ok(self, target_ms: float, stage: str = "pipeline_total") -> bool:
        """Return True if the p95 latency for *stage* is within *target_ms*."""
        st = self._stages.get(stage)
        if st is None:
            return True  # no data yet → budget not violated
        s = st.summary()
        return float(s.get("p95_ms", 0.0)) <= target_ms

    def check_latency_budget(
        self,
        logger: logging.Logger,
        target_ms: float,
        stage: str = "pipeline_total",
    ) -> None:
        """Log a warning if the last recorded pipeline latency exceeds the budget."""
        st = self._stages.get(stage)
        if st and st._samples:
            last = list(st._samples)[-1]
            if last > target_ms:
                log(logger, logging.WARNING, "latency budget exceeded",
                    stage=stage, last_ms=round(last, 1), target_ms=target_ms)
