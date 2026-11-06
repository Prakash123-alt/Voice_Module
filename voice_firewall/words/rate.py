"""Speech-rate and audio/transcript mismatch checks (Phase 3).

Normal conversational speech: 100–180 words per minute (1.67–3.0 wps).
Extreme outliers may indicate machine-generated or injected audio.
"""
from __future__ import annotations

# Thresholds (words per minute)
_WPM_MIN = 20.0    # below this → suspiciously slow / very few words
_WPM_MAX = 350.0   # above this → suspiciously fast (machine-generated)


def check_rate_mismatch(transcript: str, speech_ms: int) -> list[str]:
    """Return flags if the transcript word count is inconsistent with speech duration.

    Parameters
    ----------
    transcript:  The transcribed text (may be empty).
    speech_ms:   Duration of detected speech in milliseconds (from VAD).

    Returns
    -------
    List of flag strings (empty when no anomaly detected).
    """
    flags: list[str] = []

    if not transcript or not transcript.strip():
        # Empty/whitespace-only transcript while speech was detected.
        if speech_ms >= 300:   # ignore sub-300 ms blips
            flags.append("empty-transcript")
        return flags

    words = transcript.split()
    n_words = len(words)
    duration_min = speech_ms / 60_000.0   # ms → minutes

    if duration_min <= 0:
        return flags

    wpm = n_words / duration_min

    if wpm > _WPM_MAX:
        flags.append(f"rate-too-fast:{wpm:.0f}wpm")
    elif wpm < _WPM_MIN and n_words >= 3:
        # Only flag very-slow rate when there are enough words to be meaningful
        flags.append(f"rate-too-slow:{wpm:.0f}wpm")

    return flags
