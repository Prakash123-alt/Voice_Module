"""Orchestrator: wires the pipeline together with asyncio and the state machine.

Threads:
  * PortAudio callbacks  -> thread-safe queues (audio_io)
  * DSP thread           -> AEC -> denoise -> VAD -> segmenter (+ barge-in verification)
                            posts events into an asyncio.Queue
  * asyncio loop         -> speaker verification, STT, LLM streaming, TTS, playback, state

Barge-in: while PROCESSING or SPEAKING, the DSP thread keeps running VAD + speaker verification on
the (AEC-cleaned) signal. Once the *verified* user has spoken for >= barge_in.min_speech_ms it posts
a barge candidate; the loop verifies it off-thread, flushes playback (<= one 20 ms frame), cancels
the in-flight LLM/TTS task and goes back to LISTENING. The interrupting utterance is then handled as
the next request. Interruption is fully autonomous: there is no button, hotkey or wake word.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .chunker import SentenceChunker, clean_for_speech
from .config import Config
from .dsp import rms_db
from .interfaces import (AECProcessor, AudioInput, AudioOutput, Denoiser, LLMClient, SpeakerVerifier, STTEngine,
                         TTSEngine, VADDetector)
from .logging_utils import Metrics, log
from .speaker import cosine_similarity
from .state import State, StateMachine
from .vad import Segmenter

logger = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str                      # "segment" | "barge_candidate" | "device_error" | "stop"
    audio: Optional[np.ndarray] = None
    info: Optional[dict[str, Any]] = None


class VoiceAssistant:
    def __init__(self, cfg: Config, *, audio_in: AudioInput, audio_out: AudioOutput, aec: AECProcessor,
                 denoiser: Denoiser, vad: VADDetector, verifier: SpeakerVerifier, stt: STTEngine, llm: LLMClient,
                 tts: TTSEngine, metrics: Optional[Metrics] = None) -> None:
        self.cfg = cfg
        self.audio_in, self.audio_out, self.aec, self.denoiser, self.vad = audio_in, audio_out, aec, denoiser, vad
        self.verifier, self.stt, self.llm, self.tts = verifier, stt, llm, tts
        self.metrics = metrics or Metrics()
        # Optional observer (e.g. the web UI): on_event(kind, data). May be called from the DSP thread,
        # so it must be thread-safe and non-blocking. Exceptions in it are swallowed.
        self.on_event: Optional[Callable[[str, dict], None]] = None
        self.muted = False
        self._frame_i = 0
        self.sm = StateMachine(on_change=lambda old, new: self._emit("state", **{"from": old.value, "to": new.value}))
        self.segmenter = Segmenter(frame_ms=cfg.audio.frame_ms, threshold=cfg.vad.threshold,
                                   min_speech_ms=cfg.vad.min_speech_ms, hangover_ms=cfg.vad.hangover_ms,
                                   pad_ms=cfg.vad.pad_ms, max_segment_s=cfg.vad.max_segment_s)
        self.history: list[dict] = []
        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._dsp_thread: Optional[threading.Thread] = None
        self._respond_task: Optional[asyncio.Task] = None
        self._barge_checked_ms = -1
        self._barge_pending = threading.Event()   # a barge-in verification is in flight
        self._recovering = threading.Event()      # device recovery in progress
        self._bg_tasks: set[asyncio.Task] = set()
        self._ref_buf: deque[np.ndarray] = deque(maxlen=100)  # last ~2 s of far-end (TTS) reference
        self._frame_e: deque[tuple[float, float]] = deque(maxlen=100)  # (mic, clean) energy per frame

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.audio_out.start()
        self.audio_in.start()
        self._dsp_thread = threading.Thread(target=self._dsp_loop, name="dsp", daemon=True)
        self._dsp_thread.start()
        self.sm.transition(State.LISTENING, "start")
        log(logger, logging.INFO, "ready - speak when you like", aec="on" if self.aec.available else "gated",
            barge_in=self.cfg.barge_in.enabled and self.aec.available)
        try:
            while not self._stop.is_set():
                ev = await self._events.get()
                await self._handle(ev)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._stop.is_set() and self._dsp_thread is None:
            return
        self._stop.set()
        for t in list(self._bg_tasks):
            t.cancel()
        await self._cancel_response()
        self.audio_out.flush()
        self.audio_in.stop()
        self.audio_out.stop()
        if self._dsp_thread is not None:
            self._dsp_thread.join(timeout=2)
            self._dsp_thread = None
        await self.llm.aclose()
        self.sm.try_transition(State.IDLE, "shutdown")
        log(logger, logging.INFO, "latency summary", **{k: v["p50_ms"] for k, v in self.metrics.summary().items()})

    def stop(self) -> None:
        self._stop.set()
        if self._loop:
            self._loop.call_soon_threadsafe(self._events.put_nowait, Event("stop"))

    # ------------------------------------------------------------------ external control (UI)
    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, data)
            except Exception:
                pass

    def submit_text(self, text: str) -> bool:
        """Typed input: same pipeline as speech, minus STT/speaker check. Only accepted while LISTENING."""
        text = text.strip()
        if not text or not self.sm.is_(State.LISTENING):
            return False
        self.sm.transition(State.PROCESSING, "typed input")
        self._respond_task = asyncio.create_task(self._respond(None, text=text), name="respond")
        return True

    # ------------------------------------------------------------------ DSP thread
    def _post(self, ev: Event) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._events.put_nowait, ev)

    def _dsp_loop(self) -> None:
        idle_reads = 0
        while not self._stop.is_set():
            mic = self.audio_in.read(timeout=0.5)
            if mic is None:
                idle_reads += 1
                err = getattr(self.audio_in, "error", None)
                if (err or idle_reads >= 6) and not self._recovering.is_set():
                    self._recovering.set()
                    self._post(Event("device_error", info={"err": str(err or "no audio from mic for 3 s")}))
                    idle_reads = 0
                continue
            idle_reads = 0
            self.process_frame(mic)

    def process_frame(self, mic: np.ndarray) -> None:
        """One 20 ms frame through AEC -> denoise -> VAD -> segmenter. Runs on the DSP thread;
        must stay cheap - anything heavy (embeddings, STT) is handed to the asyncio loop."""
        bi = self.cfg.barge_in
        t0 = time.perf_counter()
        if self.muted:
            mic = np.zeros_like(mic)
        ref = self.audio_out.pop_reference()
        # keep the reference tap from lagging behind capture
        backlog = getattr(self.audio_out, "reference_backlog", lambda: 0)()
        while backlog > 5:
            extra = self.audio_out.pop_reference()
            if extra is not None:
                self._ref_buf.append(extra)
            backlog -= 1
        if ref is not None:
            self._ref_buf.append(ref)
        clean = self.denoiser.process(self.aec.process(mic, ref))
        self._frame_e.append((float(np.dot(mic, mic)), float(np.dot(clean, clean))))
        prob = self.vad.probability(clean)
        ev = self.segmenter.push(clean, prob)
        self.metrics.record("dsp_frame", (time.perf_counter() - t0) * 1000)
        self._frame_i += 1
        if self.on_event is not None and self._frame_i % 3 == 0:  # ~16 Hz level meter
            self._emit("level", mic_db=round(rms_db(mic), 1), vad=round(prob, 3))
        if ev is None:
            return
        if ev.kind == "start":
            self._barge_checked_ms = -1
        elif ev.kind == "end":
            self._post(Event("segment", audio=ev.audio, info={"speech_ms": ev.speech_ms, "end_t": time.perf_counter()}))
        elif (ev.kind == "speech" and bi.enabled and not self._barge_pending.is_set()
              # interruption is autonomous and continuous: while SPEAKING the echo must be
              # cancelled (needs the AEC); while PROCESSING nothing is playing yet, so the
              # same verified-voice check applies unconditionally
              and (self.sm.is_(State.PROCESSING) or (self.sm.is_(State.SPEAKING) and self.aec.available))
              and ev.speech_ms >= bi.min_speech_ms and ev.speech_ms - self._barge_checked_ms >= bi.min_speech_ms):
            # re-check at most every min_speech_ms while the segment grows; one check in flight at a time
            self._barge_checked_ms = ev.speech_ms
            self._barge_pending.set()
            # near-end ratio: fraction of mic energy that survives the AEC. ~1 = real person at
            # the mic, ~0 = mostly echo — the strongest "you vs echo" discriminator available
            n = min(len(self._frame_e), max(1, bi.verify_window_ms // self.cfg.audio.frame_ms))
            recent = list(self._frame_e)[-n:]
            mic_sum = sum(m for m, _ in recent)
            near_ratio = (sum(c for _, c in recent) / mic_sum) ** 0.5 if mic_sum > 0 else 0.0
            self._post(Event("barge_candidate", audio=self.segmenter.current_speech_audio(bi.verify_window_ms),
                             info={"speech_ms": ev.speech_ms, "near_ratio": round(near_ratio, 2)}))

    # ------------------------------------------------------------------ event handling
    async def _handle(self, ev: Event) -> None:
        if ev.kind == "stop":
            return
        if ev.kind == "device_error":
            # background task: recovery can take ~15 s and must not block stop/barge-in/other events
            self._spawn(self._recover_device(ev.info or {}))
        elif ev.kind == "barge_candidate":
            # verify off the event loop and off the DSP thread, without blocking other events
            self._spawn(self._check_barge_in(ev.audio, ev.info or {}))
        elif ev.kind == "segment":
            await self._on_segment(ev.audio, ev.info or {})

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    async def _check_barge_in(self, window: np.ndarray, info: dict) -> None:
        try:
            min_samples = int(self.cfg.barge_in.min_speech_ms * self.cfg.audio.sample_rate / 1000)
            if window.shape[0] < min_samples:
                return
            with self.metrics.timer("barge_verify"):
                score, accepted, echo_score = await asyncio.to_thread(
                    self._barge_verdict, window, info.get("near_ratio", 0.0))
            floor = self.cfg.barge_in.threshold
            if floor is None:
                floor = self.cfg.speaker.threshold - 0.25
            # a chip for accepted or close-call checks; sub-floor echo noise stays in the logs only
            if accepted or score >= floor:
                self._emit("speaker", accepted=accepted, score=round(score, 3),
                           threshold=self.cfg.speaker.threshold, speech_ms=info.get("speech_ms"),
                           barge=True, echo_score=echo_score, near=info.get("near_ratio"))
            if accepted:
                await self._on_barge_in({"score": round(score, 3), **info})
            else:
                log(logger, logging.INFO, "ignored interruption (not enrolled user / echo)",
                    score=round(score, 3), echo=echo_score, near=info.get("near_ratio"),
                    threshold=self.cfg.speaker.threshold)
        except Exception as e:
            log(logger, logging.WARNING, "barge-in verification failed", err=str(e))
        finally:
            self._barge_pending.clear()

    def _barge_verdict(self, window: np.ndarray, near_ratio: float = 0.0) -> tuple[float, bool, Optional[float]]:
        """Decide whether a barge-in candidate is the enrolled user. The plain cosine score must
        clear cfg.speaker.threshold. Below the bar, speech heard over residual echo (the cleaned
        window mixes the user with leftover TTS, dragging the score down) gets a second chance:
        it must clear a relaxed floor AND either match the user better than the far-end reference
        it would be echoing, or come in mostly un-attenuated through the AEC (near_ratio >= 0.5 —
        echo is cancelled ~30 dB so a high survival ratio means a real voice at the mic)."""
        profile = getattr(self.verifier, "profile", None)
        threshold = self.cfg.speaker.threshold
        floor = self.cfg.barge_in.threshold
        if floor is None:
            floor = threshold - 0.25
        if profile is None:
            res = self.verifier.verify(window, 1)
            return res.score, res.accepted, None
        cand = self.verifier.embed(window)
        score = float(cosine_similarity(cand, profile))
        if score >= threshold:
            return score, True, None
        ref = np.concatenate(list(self._ref_buf)) if self._ref_buf else np.zeros(0, dtype=np.float32)
        if ref.size == 0 or float(np.max(np.abs(ref))) < 1e-3:
            return score, False, None
        echo = round(float(cosine_similarity(cand, self.verifier.embed(ref))), 3)
        return score, score >= floor and (score > echo + 0.05 or near_ratio >= 0.5), echo

    async def _on_barge_in(self, info: dict) -> None:
        if not self.sm.is_(State.SPEAKING, State.PROCESSING):
            return
        t0 = time.perf_counter()
        was_speaking = self.sm.is_(State.SPEAKING)
        if was_speaking:
            self.sm.transition(State.INTERRUPTED, "user barge-in")
        self.audio_out.flush()                       # stop sound now (no-op while PROCESSING)
        await self._cancel_response()                # cancel LLM stream + TTS
        self.audio_out.flush()                       # anything that raced in after cancel
        stop_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.metrics.record("barge_in_stop", stop_ms)
        log(logger, logging.INFO, "interrupted", stop_ms=stop_ms, was_speaking=was_speaking, **info)
        self._emit("bargein", stop_ms=stop_ms, was_speaking=was_speaking, **info)
        self.sm.transition(State.LISTENING, "after interrupt")

    async def _on_segment(self, audio: np.ndarray, info: dict) -> None:
        if not self.sm.is_(State.LISTENING):
            log(logger, logging.DEBUG, "segment ignored (busy)", state=self.sm.state.value, **info)
            return
        if audio.shape[0] < self.cfg.speaker.min_segment_ms * self.cfg.audio.sample_rate / 1000:
            return
        with self.metrics.timer("speaker_verify"):
            res = await asyncio.to_thread(self.verifier.verify, audio)
        self._emit("speaker", accepted=res.accepted, score=round(res.score, 3), threshold=res.threshold,
                   speech_ms=info.get("speech_ms"))
        if not res.accepted:
            log(logger, logging.INFO, "rejected speaker", score=round(res.score, 3), threshold=res.threshold,
                speech_ms=info.get("speech_ms"))
            return
        if not self.sm.is_(State.LISTENING):  # state may have changed while verifying
            return
        log(logger, logging.INFO, "accepted speaker", score=round(res.score, 3), speech_ms=info.get("speech_ms"))
        self.sm.transition(State.PROCESSING, "verified utterance")
        self._respond_task = asyncio.create_task(self._respond(audio, info.get("end_t")), name="respond")

    async def _cancel_response(self) -> None:
        t = self._respond_task
        if t and not t.done():
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._respond_task = None

    async def _recover_device(self, info: dict) -> None:
        log(logger, logging.ERROR, "audio device problem - trying to restart input", **info)
        try:
            self.audio_in.stop()
            for attempt in range(5):
                await asyncio.sleep(1 + attempt)
                try:
                    self.audio_in.start()
                except Exception as e:
                    log(logger, logging.WARNING, "mic restart failed", attempt=attempt + 1, err=str(e))
                    continue
                for comp in (self.aec, self.denoiser, self.vad, self.segmenter):
                    comp.reset()
                log(logger, logging.INFO, "mic restarted", attempt=attempt + 1)
                return
            log(logger, logging.CRITICAL, "giving up on audio input")
            self.stop()
        finally:
            self._recovering.clear()

    # ------------------------------------------------------------------ response pipeline
    def _messages(self, user_text: str) -> list[dict]:
        hist = self.history[-2 * self.cfg.llm.history_turns:]
        return [{"role": "system", "content": self.cfg.llm.system_prompt}, *hist, {"role": "user", "content": user_text}]

    def _remember(self, user: str, assistant: str) -> None:
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        del self.history[:-2 * self.cfg.llm.history_turns]

    async def _respond(self, audio: Optional[np.ndarray], end_t: Optional[float] = None,
                       text: Optional[str] = None) -> None:
        end_t = end_t or time.perf_counter()
        first_audio_logged = False
        reply_parts: list[str] = []
        remembered = False
        speaker: Optional[asyncio.Task] = None
        turn: dict[str, Any] = {"source": "text" if text is not None else "voice"}
        try:
            if text is None:
                t_stt = time.perf_counter()
                with self.metrics.timer("stt"):
                    text = await self.stt.transcribe(audio)
                turn["stt_ms"] = round((time.perf_counter() - t_stt) * 1000)
            if not text.strip():
                log(logger, logging.INFO, "empty transcript")
                self.sm.transition(State.LISTENING, "empty transcript")
                return
            log(logger, logging.INFO, "user", text=text)
            self._emit("transcript", role="user", text=text, source=turn["source"])
            sentences: asyncio.Queue[Optional[str]] = asyncio.Queue()

            async def speak_worker() -> None:
                nonlocal first_audio_logged
                while (s := await sentences.get()) is not None:
                    if not (s := clean_for_speech(s)):
                        continue
                    t_tts = time.perf_counter()
                    with self.metrics.timer("tts_sentence"):
                        # aclosing: on cancel the TTS generator (and its worker/request) is closed immediately
                        async with aclosing(self.tts.synthesize(s)) as chunks:
                            async for chunk in chunks:
                                if not first_audio_logged:
                                    first_audio_logged = True
                                    e2e = (time.perf_counter() - end_t) * 1000
                                    self.metrics.record("e2e_first_audio", e2e)
                                    log(logger, logging.INFO, "first audio", e2e_ms=round(e2e))
                                    turn.update(e2e_ms=round(e2e), tts_ms=round((time.perf_counter() - t_tts) * 1000))
                                    self._emit("turn", **turn)
                                    self.sm.try_transition(State.SPEAKING, "first tts chunk")
                                self.audio_out.play(chunk)

            speaker = asyncio.create_task(speak_worker(), name="speak")
            chunker = SentenceChunker(min_chars=self.cfg.tts.min_chunk_chars)
            t_llm = time.perf_counter()
            first_token = None
            try:
                # aclosing: on barge-in the SSE response is closed now, not whenever the GC runs
                async with aclosing(self.llm.stream_chat(self._messages(text), model=self.cfg.llm.model,
                                                         max_tokens=self.cfg.llm.max_tokens,
                                                         temperature=self.cfg.llm.temperature,
                                                         **self.cfg.llm.extra_body)) as stream:
                    async for delta in stream:
                        if first_token is None:
                            first_token = time.perf_counter()
                            self.metrics.record("llm_first_token", (first_token - t_llm) * 1000)
                            turn["llm_first_ms"] = round((first_token - t_llm) * 1000)
                        reply_parts.append(delta)
                        self._emit("reply_delta", text=delta)
                        for s in chunker.feed(delta):
                            await sentences.put(s)
                for s in chunker.flush():
                    await sentences.put(s)
                self.metrics.record("llm_total", (time.perf_counter() - t_llm) * 1000)
            finally:
                sentences.put_nowait(None)
            await speaker
            reply = "".join(reply_parts).strip()
            log(logger, logging.INFO, "assistant", text=reply)
            self._emit("transcript", role="assistant", text=reply, interrupted=False)
            if reply:
                self._remember(text, reply)
                remembered = True
            await self.audio_out.wait_until_done()
            if self.sm.is_(State.PROCESSING, State.SPEAKING):
                self.sm.transition(State.LISTENING, "response done")
        except asyncio.CancelledError:
            # keep the (partial) answer so the conversation stays coherent, marked as cut off
            if remembered:
                self.history[-1]["content"] += " [interrupted by user]"
            elif reply_parts and text:
                self._remember(text, "".join(reply_parts) + " [interrupted by user]")
            self._emit("interrupted_reply", text="".join(reply_parts).strip())
            raise
        except Exception as e:
            log(logger, logging.ERROR, "response failed", err=f"{e.__class__.__name__}: {e}")
            self._emit("error", message=f"Response failed: {e.__class__.__name__}: {e}")
            self.audio_out.flush()
            if not self.sm.is_(State.LISTENING):
                self.sm.try_transition(State.LISTENING, "error")
        finally:
            if speaker is not None and not speaker.done():
                speaker.cancel()
                try:
                    await speaker
                except (asyncio.CancelledError, Exception):
                    pass
