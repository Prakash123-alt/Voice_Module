"""Optional SpeakerLock isolation stage (speaker_lock package).

Runs target-speaker extraction plus the rolling verification gate on a worker thread, so the
DSP thread only enqueues mic frames and dequeues gated frames. Enable via `isolation:` in
config.yaml. Requires a SpeakerLock enrollment (`python -m speakerlock.cli enroll --user me
--file <clip>`), which is separate from the Resemblyzer `profile/voice.npy`.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional

import numpy as np

from .logging_utils import log

logger = logging.getLogger(__name__)


class IsolationStage:
    """push() is DSP-thread safe (enqueue only); pop() returns the next gated frame or None.

    SpeakerLock emits output delayed ~60-100 ms behind its input, so pop() may return None
    while the worker catches up; the caller feeds silence then and the gated stream simply
    lags the mic by the pipeline delay.
    """

    def __init__(self, frame_samples: int, user_id: str, store_dir: str) -> None:
        from speakerlock import Config as SLConfig      # noqa: PLC0415
        from speakerlock.pipeline import SpeakerLock   # noqa: PLC0415
        from speakerlock.stream import LiveRunner      # noqa: PLC0415
        cfg = SLConfig()
        cfg.store_dir = store_dir
        self._lock = SpeakerLock.create(cfg, user_id, store_dir,
                                        on_unverified=lambda i: log(
                                            logger, logging.INFO, "isolation: unverified speaker",
                                            best_user=i.get("best_user"), score=round(i.get("score") or 0, 3)))
        self._frame = frame_samples
        self._frames: deque[np.ndarray] = deque()
        self._tail = np.zeros(0, np.float32)
        self.meta = None                               # latest speakerlock ChunkMeta
        self._lock_sink = threading.Lock()
        self._runner = LiveRunner(self._lock, self._sink).start()

    def _sink(self, pcm: np.ndarray, meta) -> None:      # worker thread
        with self._lock_sink:
            self.meta = meta
            self._tail = np.concatenate([self._tail, pcm])
            n = self._tail.shape[0] - self._tail.shape[0] % self._frame
            for i in range(0, n, self._frame):
                self._frames.append(self._tail[i:i + self._frame])
            self._tail = self._tail[n:]

    @property
    def verified(self) -> bool:
        """True while the gate is open for the enrolled user (best-effort, worker-thread value)."""
        return bool(self.meta and self.meta.verified)

    def push(self, frame: np.ndarray) -> None:
        self._runner.push(frame)

    def pop(self) -> Optional[np.ndarray]:
        with self._lock_sink:
            return self._frames.popleft() if self._frames else None

    def close(self) -> None:
        self._runner.stop(flush=False)
        self._lock.close()


def build_isolation(cfg) -> Optional[IsolationStage]:
    """Create the stage from cfg.isolation; returns None (with a warning) if disabled, the
    package is missing, or the user is not enrolled."""
    iso = getattr(cfg, "isolation", None)
    if iso is None or not iso.enabled:
        return None
    try:
        stage = IsolationStage(int(cfg.audio.frame_ms * cfg.audio.sample_rate / 1000),
                               iso.user_id, iso.store_dir)
    except ImportError:
        log(logger, logging.WARNING, "isolation: speakerlock package not importable "
            "(pip install -e speaker_lock); running without isolation")
        return None
    except Exception as e:
        log(logger, logging.WARNING, "isolation: disabled", err=str(e),
            hint="enroll first: python -m speakerlock.cli enroll --user <id> --file <clip>")
        return None
    log(logger, logging.INFO, "isolation: speaker-locked gate active", user=iso.user_id)
    return stage
