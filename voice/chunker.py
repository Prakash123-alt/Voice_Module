"""Sentence-level chunker: feed streamed token deltas, get back complete sentences as
soon as they close, so TTS can start before the LLM finishes."""
from __future__ import annotations

import re
from typing import Iterator

_MD_RULES = [
    (re.compile(r"```.*?```", re.S), " "),                  # code blocks
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),           # [text](url) -> text
    (re.compile(r"https?://\S+"), " link "),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),            # headings
    (re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M), ""),    # list markers
    (re.compile(r"[*_`~]{1,3}"), ""),                        # emphasis / inline code
    (re.compile(r"\s+"), " "),
]


def clean_for_speech(text: str) -> str:
    """Strip markdown so TTS doesn't read out asterisks, hashes or URLs."""
    for rx, rep in _MD_RULES:
        text = rx.sub(rep, text)
    return text.strip()


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "inc", "ltd", "fig"}
_END = re.compile(r"([.!?]+[\"'”’)\]]*)(\s+|$)")


class SentenceChunker:
    def __init__(self, min_chars: int = 12, max_chars: int = 300) -> None:
        self.min_chars, self.max_chars = min_chars, max_chars
        self._buf = ""

    @property
    def pending(self) -> str:
        return self._buf

    def _is_boundary(self, text: str, end: int) -> bool:
        head = text[:end].rstrip()
        # Last token before the punctuation
        m = re.search(r"(\S+)$", head)
        if not m:
            return True
        word = m.group(1).rstrip(".!?\"'”’)]").lower()
        if word in _ABBREV or len(word) == 1 and word.isalpha():  # "e.g." / initials like "J."
            return False
        # Decimal number like "3." followed by digit is handled by \s+ requirement; "1." list items pass.
        return True

    def feed(self, delta: str) -> Iterator[str]:
        self._buf += delta
        while True:
            found = None
            for m in _END.finditer(self._buf):
                end = m.end(1)
                if end < self.min_chars and len(self._buf) > end:
                    continue  # too short to be worth a TTS call; keep accumulating
                if m.group(2) == "" and len(self._buf) == end:
                    break  # punctuation at the very end of buffer: wait for a following space/next token
                if self._is_boundary(self._buf, end):
                    found = end
                    break
            if found is None:
                if len(self._buf) >= self.max_chars:  # no punctuation for a long time: cut at last space
                    cut = self._buf.rfind(" ", 0, self.max_chars)
                    cut = cut if cut > 0 else self.max_chars
                    out, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
                    if out:
                        yield out
                    continue
                return
            out, self._buf = self._buf[:found].strip(), self._buf[found:].lstrip()
            if out:
                yield out

    def flush(self) -> Iterator[str]:
        out, self._buf = self._buf.strip(), ""
        if out:
            yield out
