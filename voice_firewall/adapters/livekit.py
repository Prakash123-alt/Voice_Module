"""LiveKit integration adapter (Phase 4).

Wraps the Voice Firewall as a LiveKit room audio processor.
LiveKit delivers audio via track subscriptions; this adapter bridges
Firewall.process() into that model.

Install LiveKit::
    pip install livekit

Usage::
    from voice_firewall.adapters.livekit import LiveKitFirewallAdapter
    from voice_firewall._config import FirewallConfig

    adapter = LiveKitFirewallAdapter(FirewallConfig.from_yaml())
    await adapter.connect(url="ws://localhost:7880", token="...")
    # adapter streams mic audio through the firewall automatically

Docs: https://docs.livekit.io/
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from voice_firewall._config import FirewallConfig
from voice_firewall._pipeline import Firewall
from voice_firewall._types import TaggedSegment

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 320   # 20 ms at 16 kHz


class LiveKitFirewallAdapter:
    """Streams mic audio from a LiveKit room through the Voice Firewall.

    Audio from every ``RemoteAudioTrack`` subscription is accumulated in a
    ring buffer, segmented by the firewall's internal VAD, and passed through
    all guards.  Passed segments are forwarded to ``on_segment_passed``.

    If LiveKit is not installed the class imports cleanly but ``connect``
    raises ``ImportError``.
    """

    def __init__(
        self,
        cfg: FirewallConfig,
        on_passed: Optional[callable] = None,
        on_dropped: Optional[callable] = None,
    ) -> None:
        self._fw = Firewall(cfg)
        self._on_passed = on_passed
        self._on_dropped = on_dropped
        self._buf: list[np.ndarray] = []

    async def connect(self, url: str, token: str) -> None:
        """Connect to a LiveKit room and start processing audio."""
        try:
            import livekit  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "livekit is not installed. Run: pip install livekit"
            ) from exc

        self._fw.start()
        room = livekit.Room()

        @room.on("track_subscribed")
        def on_track(track, publication, participant):
            if isinstance(track, livekit.AudioTrack):
                asyncio.ensure_future(self._stream_track(track))

        await room.connect(url, token)
        logger.info("LiveKitFirewallAdapter: connected to %s", url)

    async def _stream_track(self, track: object) -> None:
        """Read audio frames from a LiveKit AudioTrack and process them."""
        async for frame in track:  # type: ignore[attr-defined]
            # LiveKit AudioFrame: frame.data is bytes, frame.sample_rate, frame.num_channels
            audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            if getattr(frame, "num_channels", 1) > 1:
                audio = audio.reshape(-1, frame.num_channels).mean(axis=1)
            # Feed 20 ms frames; accumulate a 1-second chunk then process
            self._buf.append(audio[:_FRAME_SAMPLES])
            if len(self._buf) >= 50:   # ~1 second
                segment = np.concatenate(self._buf)
                self._buf.clear()
                tagged = self._fw.process(segment)
                if tagged is None:
                    if self._on_dropped:
                        self._on_dropped("dropped by guard pipeline")
                else:
                    if self._on_passed:
                        self._on_passed(tagged)

    def stop(self) -> None:
        self._fw.stop()

    @property
    def firewall(self) -> Firewall:
        return self._fw
