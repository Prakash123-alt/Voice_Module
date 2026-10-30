"""One WebSocket connection = one Session. Audio in (binary, int16 16 kHz) ->
interrupt manager + segmenter -> verified turns -> STT -> LLM stream -> TTS ->
audio out (binary). All control events are JSON text frames.

Interruption is automatic everywhere: the interrupt manager sees every frame;
on CONFIRMED we emit `interrupted`, cancel TurnContext (SSE, TTS, queues) and
truncate history at played_ms so the model knows what was actually heard.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from voice.chunker import SentenceChunker, clean_for_speech
from voice.state import State, StateMachine
from voice.vad import Segmenter

from .interrupt import InterruptConfig, InterruptManager

logger = logging.getLogger(__name__)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))
_turn_ids = itertools.count(1)


class TurnContext:
    """One assistant turn: LLM stream -> sentence chunker -> TTS -> audio out."""
    def __init__(self, session: "Session", user_text: str) -> None:
        self.session, self.user_text = session, user_text
        self.id = next(_turn_ids)
        self.cancelled = False
        self.reply_parts: list[str] = []
        self.sent_marks: list[tuple[float, int]] = []   # (audio_ms_cum, chars_cum)
        self.sent_ms = 0.0

    def cancel(self) -> None:
        self.cancelled = True

    def spoken_so_far(self, played_ms: float) -> str:
        """Truncate the reply to what was actually played, per-sentence timing."""
        chars = 0
        for ms_cum, chars_cum in self.sent_marks:
            if ms_cum > played_ms:
                break
            chars = chars_cum
        return "".join(self.reply_parts)[:chars].strip()


class Session:
    def __init__(self, cfg, stt, llm, tts, vad, verifier, segmenter: Segmenter,
                 send_json, send_bytes, icfg: Optional[InterruptConfig] = None) -> None:
        self.cfg, self.stt, self.llm, self.tts = cfg, stt, llm, tts
        self.vad, self.verifier, self.segmenter = vad, verifier, segmenter
        self.send_json, self.send_bytes = send_json, send_bytes
        self.im = InterruptManager(icfg or InterruptConfig(),
                                   has_profile=verifier is not None)
        self.sm = StateMachine(on_change=lambda o, n: self._emit("state", to=n.value))
        self.history: list[dict] = []
        self._ref_rms: deque[float] = deque(maxlen=100)   # RMS of frames we're sending out
        self._seg_start_ms = 0
        self.turn: Optional[TurnContext] = None
        self._tasks: set[asyncio.Task] = set()
        self._verify_pending = False
        self._events: asyncio.Queue = asyncio.Queue()
        self._played_ms = 0.0
        self.closed = False
        self._loop = asyncio.get_running_loop()
        # timestamps: server-side frame index -> monotonic ms
        self._frame_i = 0

    # ------------------------------------------------------------- WS-facing API
    async def run(self) -> None:
        self.sm.transition(State.LISTENING, "session start")
        while not self.closed:
            ev = await self._events.get()
            await self._handle(ev)

    def feed_audio(self, pcm: bytes) -> None:
        """Binary mic frame from the client (already AEC-cleaned in browser)."""
        self._rx = getattr(self, "_rx", 0) + 1
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self._rx == 1 or self._rx % 100 == 0:
            logger.info("mic frame %d rms=%.4f", self._rx, _rms(audio))
        for off in range(0, len(audio), 320):                      # 20 ms frames
            f = audio[off:off + 320]
            if f.shape[0] == 320:
                self._frame_i += 1
                self._process_frame(f)

    def playback_progress(self, played_ms: float) -> None:
        self._played_ms = played_ms

    def feed_control(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "played":
            self._played_ms = float(msg.get("played_ms", 0))
        elif t == "text":
            self._spawn(self._user_text(msg.get("text", "")))

    def close(self) -> None:
        self.closed = True
        if self.turn:
            self.turn.cancel()
        for t in list(self._tasks):
            t.cancel()

    # ------------------------------------------------------------- frame path
    def _process_frame(self, f: np.ndarray) -> None:
        mic_rms = _rms(f)
        prob = self.vad.probability(f)
        ref = self._ref_rms[-1] if self._ref_rms else 0.0
        ev = self.segmenter.push(f, prob)
        act = self.im.update(prob, mic_rms, ref)
        if act == "duck":
            self._emit("duck", level=0.2)
        elif act == "restore":
            self._emit("restore")
        elif act == "speech_started" or (act == "duck"):
            self._emit("speech_started")
        elif act == "verify" and not self._verify_pending:
            self._verify_pending = True
            win_ms = min(self.im._speech_ms, self.im.cfg.verify_max_ms)
            audio = self.segmenter.current_speech_audio(win_ms)
            self._spawn(self._verify_interrupt(audio, self.im._speech_ms))
        if ev is not None and ev.kind == "end":
            self._post(("segment", ev))

    async def _verify_interrupt(self, window: np.ndarray, speech_ms: int) -> None:
        try:
            if self.verifier is None:
                accepted, score = True, 1.0               # speaker lock off
            else:
                min_samples = int(0.25 * self.cfg.audio.sample_rate)
                res = await asyncio.to_thread(self.verifier.verify, window, min_samples)
                score = res.score
                accepted = score >= self.im.score_threshold(min(speech_ms, 800))
            out = self.im.verify_result(accepted, speech_ms)
            self._emit("speaker", accepted=accepted, score=round(score, 3), barge=True)
            if out == "confirmed":
                await self._interrupt(speech_ms, score)
            elif out == "restore":
                self._emit("restore")
        except Exception as e:
            logger.warning("interrupt verify failed: %s", e)
        finally:
            self._verify_pending = False

    async def _interrupt(self, speech_ms: int, score: float) -> None:
        """CONFIRMED: hard stop playback + cancel turn, <100 ms target."""
        t0 = time.perf_counter()
        spoken = self.turn.spoken_so_far(self._played_ms) if self.turn else ""
        if self.turn and spoken:
            self._remember(self.turn.user_text, spoken + " [interrupted]")
        if self.turn:
            self.turn.cancel()
        self.im.confirmed_stopped()
        self._emit("interrupted", turn_id=self.turn.id if self.turn else 0,
                   played_ms=round(self._played_ms), score=round(score, 3))
        if self.sm.is_(State.SPEAKING, State.PROCESSING):
            self.sm.try_transition(State.INTERRUPTED, "barge-in") or \
                self.sm.try_transition(State.LISTENING, "barge-in")
        self.sm.try_transition(State.LISTENING, "after interrupt")
        logger.info("interrupted stop_ms=%.1f score=%.3f", (time.perf_counter() - t0) * 1000, score)

    # ------------------------------------------------------------- turns
    async def _handle(self, ev) -> None:
        kind, data = ev
        if kind == "segment":
            await self._on_segment(data.audio, data.speech_ms)

    async def _on_segment(self, audio: np.ndarray, speech_ms: int) -> None:
        if self.sm.is_(State.PROCESSING):
            # user kept talking while we thought: cancel pending, take the merge
            if self.turn:
                self.turn.cancel()
            self.sm.try_transition(State.LISTENING, "merge")
        if not self.sm.is_(State.LISTENING):
            return
        if self.verifier is not None:
            res = await asyncio.to_thread(self.verifier.verify, audio)
            self._emit("speaker", accepted=res.accepted, score=round(res.score, 3))
            if not res.accepted:
                logger.info("rejected speaker score=%.3f", res.score)
                return
        self.sm.transition(State.PROCESSING, "verified utterance")
        text = await self.stt.transcribe(audio)
        if not text.strip():
            self.sm.try_transition(State.LISTENING, "empty transcript")
            return
        self._emit("transcript", role="user", text=text, final=True)
        await self._answer(text)

    async def _user_text(self, text: str) -> None:
        if not text.strip() or not self.sm.is_(State.LISTENING):
            return
        self.sm.transition(State.PROCESSING, "typed input")
        self._emit("transcript", role="user", text=text, final=True)
        await self._answer(text)

    async def _answer(self, text: str) -> None:
        turn = TurnContext(self, text)
        self.turn = turn
        self.im.playback_started()
        try:
            chunker = SentenceChunker(min_chars=self.cfg.tts.min_chunk_chars)
            hist = self.history[-2 * self.cfg.llm.history_turns:]
            msgs = [{"role": "system", "content": self.cfg.llm.system_prompt}, *hist,
                    {"role": "user", "content": text}]
            first = True
            async for delta in self.llm.stream_chat(msgs, model=self.cfg.llm.model,
                                                    max_tokens=self.cfg.llm.max_tokens,
                                                    temperature=self.cfg.llm.temperature,
                                                    **self.cfg.llm.extra_body):
                if turn.cancelled:
                    return
                turn.reply_parts.append(delta)
                self._emit("reply_delta", text=delta)
                for s in chunker.feed(delta):
                    await self._speak(turn, s)
            for s in chunker.flush():
                await self._speak(turn, s)
            if not turn.cancelled:
                reply = "".join(turn.reply_parts).strip()
                self._emit("transcript", role="assistant", text=reply, final=True)
                self._remember(text, reply)
                self._emit("turn_complete", turn_id=turn.id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("turn failed: %s", e)
            self._emit("error", message=str(e))
        finally:
            self.im.playback_stopped()
            if self.sm.is_(State.PROCESSING, State.SPEAKING):
                self.sm.try_transition(State.LISTENING, "turn done")

    async def _speak(self, turn: TurnContext, sentence: str) -> None:
        s = clean_for_speech(sentence)
        if not s or turn.cancelled:
            return
        chars_before = sum(len(p) for p in turn.reply_parts) - len(s)
        async for chunk in self.tts.synthesize(s):
            if turn.cancelled:
                return
            pcm = np.clip(chunk, -1, 1)
            self._ref_rms.append(_rms(pcm))
            self.send_bytes((pcm * 32767).astype(np.int16).tobytes())
            if not self.sm.is_(State.SPEAKING):
                self.sm.try_transition(State.SPEAKING, "first tts chunk")
            turn.sent_ms += len(pcm) / 16.0                    # 16 kHz -> ms
            turn.sent_marks.append((turn.sent_ms, chars_before + len(s)))

    # ------------------------------------------------------------- helpers
    def _emit(self, kind: str, **data) -> None:
        try:
            self.send_json({"type": kind, "ts": time.time(), **data})
        except Exception:
            pass

    def _remember(self, user: str, assistant: str) -> None:
        self.history += [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]
        del self.history[:-2 * self.cfg.llm.history_turns]

    def _post(self, ev) -> None:
        self._events.put_nowait(ev)

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
