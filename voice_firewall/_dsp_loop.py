"""Real-time DSP pipeline: Mic → AEC → VAD → Segmenter → RawSegment queue.

The DSPLoop runs in a background thread.  When the VAD/Segmenter declares an
utterance complete it emits a ``RawSegment`` with ``aec_suppression_db`` set
(or None when the loopback backend is unavailable), then calls
``on_segment(segment)`` — typically ``Firewall.process()``.

Usage::

    from voice_firewall._dsp_loop import DSPLoop
    from voice_firewall._pipeline import Firewall
    from voice_firewall._config import FirewallConfig

    cfg = FirewallConfig.from_yaml()
    fw  = Firewall(cfg)
    fw.start()

    dsp = DSPLoop(cfg, on_segment=fw.process)
    dsp.start()
    # ... run until done ...
    dsp.stop()
    fw.stop()

For testing without hardware, inject fake backends::

    dsp = DSPLoop(cfg, on_segment=collector,
                  mic=FakeMicInput(), loopback=StubLoopback(),
                  aec=NoAEC(), vad=EnergyVAD())
    # Then drive it manually:
    dsp.feed_frame(mic_frame, ref_frame)
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

from voice_firewall._config import FirewallConfig
from voice_firewall._logging import log
from voice_firewall._types import RawSegment
from voice_firewall.audio.aec import AECProcessor, NoAEC, build_aec
from voice_firewall.audio.dsp import rms_db
from voice_firewall.audio.loopback import LoopbackCapture, StubLoopback, build_loopback
from voice_firewall.audio.mic import AudioInput
from voice_firewall.vad.segmenter import EnergyVAD, Segmenter, build_vad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPLoop
# ---------------------------------------------------------------------------

class DSPLoop:
    """Real-time mic → AEC → VAD → Segmenter pipeline.

    ``on_segment`` is called from the DSP thread — keep it short or hand off
    to a queue/asyncio loop.  Never do heavy work (embedding, STT) on this thread.
    """

    def __init__(
        self,
        cfg: FirewallConfig,
        on_segment: Callable[[RawSegment], None],
        *,
        mic: Optional[AudioInput] = None,
        loopback: Optional[LoopbackCapture] = None,
        aec: Optional[AECProcessor] = None,
        vad: Optional[object] = None,
    ) -> None:
        self.cfg = cfg
        self._on_segment = on_segment

        acfg = cfg.audio
        scfg = cfg.guards.self_sound
        frame = acfg.frame_samples

        self._mic = mic
        self._loopback = loopback if loopback is not None else build_loopback(
            scfg.loopback_backend, sample_rate=acfg.sample_rate, frame_samples=frame
        )
        self._aec = aec if aec is not None else build_aec(
            scfg.aec_backend, acfg.sample_rate, frame
        )
        self._vad = vad if vad is not None else build_vad(acfg.sample_rate)
        self._segmenter = Segmenter(frame_ms=acfg.frame_ms)

        # Per-segment energy accumulators for suppression_db
        self._pre_sum: float = 0.0
        self._post_sum: float = 0.0
        self._n_frames: int = 0

        self._thread: Optional[threading.Thread] = None
        self._stopping = False

    # ------------------------------------------------------------------
    # Manual frame feed (used in tests without hardware)
    # ------------------------------------------------------------------

    def feed_frame(
        self,
        mic_frame: np.ndarray,
        ref_frame: Optional[np.ndarray] = None,
    ) -> None:
        """Process one 20 ms frame.  Safe to call from any thread."""
        # Energy before AEC
        pre_sq = float(np.mean(np.square(mic_frame)))

        aec_frame = self._aec.process(mic_frame, ref_frame)

        # Energy after AEC
        post_sq = float(np.mean(np.square(aec_frame)))

        # Accumulate for the segment
        self._pre_sum += pre_sq
        self._post_sum += post_sq
        self._n_frames += 1

        # VAD probability on AEC-cleaned frame
        prob = self._vad.probability(aec_frame)
        event = self._segmenter.push(aec_frame, prob)

        if event and event.kind == "end" and event.audio is not None:
            self._emit_segment(event)

    def _emit_segment(self, event: object) -> None:
        # Compute segment-level AEC suppression in dB
        loopback_available = self._loopback.available if self._loopback else False
        if loopback_available and self._n_frames > 0 and self._pre_sum > 1e-12:
            suppression_db: Optional[float] = float(
                10.0 * np.log10(self._pre_sum / (self._post_sum + 1e-12))
            )
        else:
            # Loopback not active → cannot determine suppression
            suppression_db = None

        seg = RawSegment(
            audio=event.audio,
            speech_ms=event.speech_ms,
            aec_suppression_db=suppression_db,
        )

        # Reset accumulators for the next segment
        self._pre_sum = 0.0
        self._post_sum = 0.0
        self._n_frames = 0

        log(logger, logging.DEBUG, "segment emitted",
            speech_ms=seg.speech_ms,
            aec_suppression_db=(
                round(suppression_db, 1) if suppression_db is not None else None
            ))

        self._on_segment(seg)

    # ------------------------------------------------------------------
    # Background thread (hardware mode)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._mic is None:
            raise RuntimeError("No mic backend provided; use mic= kwarg or build_mic_input().")
        self._stopping = False
        self._mic.start()
        self._loopback.start()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vfw-dsp"
        )
        self._thread.start()
        log(logger, logging.INFO, "DSPLoop started",
            aec=type(self._aec).__name__,
            loopback=type(self._loopback).__name__,
            loopback_available=self._loopback.available)

    def stop(self) -> None:
        self._stopping = True
        if self._mic is not None:
            self._mic.stop()
        if self._loopback is not None:
            self._loopback.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log(logger, logging.INFO, "DSPLoop stopped")

    def reset(self) -> None:
        """Reset VAD and segmenter state (e.g. after a barge-in)."""
        self._vad.reset()
        self._segmenter.reset()
        self._pre_sum = 0.0
        self._post_sum = 0.0
        self._n_frames = 0

    def _run(self) -> None:
        """Background thread loop."""
        assert self._mic is not None
        while not self._stopping:
            frame = self._mic.read(timeout=0.05)
            if frame is None:
                if getattr(self._mic, "error", None):
                    log(logger, logging.ERROR, "mic device error — stopping DSP",
                        err=str(self._mic.error))
                    break
                continue
            ref = self._loopback.read(timeout=0)  # non-blocking; None if no data
            self.feed_frame(frame, ref)
