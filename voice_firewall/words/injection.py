"""Injection-pattern detection for transcript text (Phase 3).

Checks transcripts for phrasing that attempts to redirect, override, or hijack
the AI agent's behaviour.  Returns a list of flag strings — never drops audio.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled injection patterns
# ---------------------------------------------------------------------------
# Each entry is (flag_name, compiled_pattern).
# flag_name is added to transcript_flags when the pattern matches.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "injection:ignore-instructions",
        re.compile(r"ignore\s+(previous|all|your|any)\s+(instructions?|commands?|prompt)",
                   re.IGNORECASE),
    ),
    (
        "injection:new-context",
        re.compile(r"new\s+context\s*:", re.IGNORECASE),
    ),
    (
        "injection:system-prompt",
        re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
    ),
    (
        "injection:you-are-now",
        re.compile(r"you\s+are\s+now\s+(a\b|an\b|the\b)", re.IGNORECASE),
    ),
    (
        "injection:forget",
        re.compile(r"forget\s+(everything|all\s+previous|your\s+instructions?)", re.IGNORECASE),
    ),
    (
        "injection:disregard",
        re.compile(r"disregard\s+(all|any|previous|your)\s+\w+", re.IGNORECASE),
    ),
    (
        "injection:act-as",
        re.compile(r"\bact\s+as\s+(a\b|an\b|if\b)", re.IGNORECASE),
    ),
    (
        "injection:pretend",
        re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    ),
    (
        "injection:jailbreak",
        re.compile(r"\bjailbreak\b", re.IGNORECASE),
    ),
    (
        "injection:roleplay",
        re.compile(r"\broleplay\s+as\b", re.IGNORECASE),
    ),
    (
        "injection:prompt-injection",
        re.compile(r"\bprompt\s+injection\b", re.IGNORECASE),
    ),
    (
        "injection:markdown-headers",
        # Markdown-style headers/instruction blocks in spoken text
        re.compile(r"#+\s*(system|human|assistant|user|instruction)\s*:", re.IGNORECASE),
    ),
    (
        "injection:inst-tag",
        re.compile(r"\[INST\]", re.IGNORECASE),
    ),
    (
        "injection:override-context",
        re.compile(r"(override|replace|reset)\s+(your\s+)?(context|instructions?|directives?)",
                   re.IGNORECASE),
    ),
]


def check_injection(transcript: str) -> list[str]:
    """Return a list of injection flag strings found in *transcript*.

    Returns an empty list if no patterns match.  Call sites should append
    these to ``TaggedSegment.transcript_flags``.
    """
    if not transcript or not transcript.strip():
        return []
    flags = []
    for flag_name, pattern in _PATTERNS:
        if pattern.search(transcript):
            flags.append(flag_name)
    return flags
