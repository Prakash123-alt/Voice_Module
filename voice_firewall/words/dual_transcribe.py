"""Dual-transcription check interface and stub (Phase 3).

When ``dual_transcribe_on_high_risk`` is enabled, the firewall transcribes a
segment twice with different settings and compares the results.  A large
divergence raises a ``dual-transcribe-mismatch`` flag.

The comparison uses normalised edit distance (Levenshtein / character level).
Full STT integration is wired by the caller; this module only contains the
comparison math and the stub interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Edit distance (no external dependencies)
# ---------------------------------------------------------------------------

def normalised_edit_distance(a: str, b: str) -> float:
    """Return character-level normalised edit distance in [0, 1].

    0.0 = identical; 1.0 = completely different.
    Uses the standard DP Levenshtein algorithm.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    # Keep only two rows to bound memory.
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n] / max(m, n)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class DualTranscriber(ABC):
    """Abstract dual-transcription provider.

    Integrations plug in a real STT backend by subclassing this and passing
    the instance to ``WordsGuard``.
    """

    @abstractmethod
    def transcribe_pair(self, audio_float32_16k: "np.ndarray") -> tuple[str, str]:
        """Return two independent transcriptions of the audio.

        Returns
        -------
        (transcript_a, transcript_b)
        """


class StubDualTranscriber(DualTranscriber):
    """Stub — always returns two identical empty transcripts (no divergence)."""

    def transcribe_pair(self, audio_float32_16k: "np.ndarray") -> tuple[str, str]:
        return "", ""


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def check_dual_transcribe(
    transcriber: DualTranscriber,
    audio: "np.ndarray",
    divergence_threshold: float = 0.25,
) -> list[str]:
    """Run dual transcription and return flags if divergence is too high.

    Parameters
    ----------
    transcriber:          A DualTranscriber implementation.
    audio:                Float32 16 kHz audio array.
    divergence_threshold: Normalised edit distance above which the flag is raised.

    Returns
    -------
    List with ``"dual-transcribe-mismatch"`` if divergence >= threshold, else [].
    """
    try:
        t_a, t_b = transcriber.transcribe_pair(audio)
        dist = normalised_edit_distance(t_a, t_b)
        if dist >= divergence_threshold:
            return [f"dual-transcribe-mismatch:{dist:.2f}"]
    except Exception:  # noqa: BLE001
        return ["dual-transcribe-error"]
    return []
