"""End-to-end orchestration with fakes: verified-speaker gating, barge-in, and ignoring other voices."""
import asyncio
import time

import numpy as np
import pytest

from voice.assistant import VoiceAssistant
from voice.config import Config
from voice.interfaces import (AECProcessor, AudioInput, AudioOutput, Denoiser, LLMClient, SpeakerVerifier, STTEngine,
                              TTSEngine, VADDetector, VerificationResult)
from voice.speaker import cosine_similarity
from voice.state import State

FS = 320  # 20 ms @ 16 kHz
USER, OTHER = 0.5, 0.2  # synthetic "voices": constant-valued frames


def frames(value, n):
    return [np.full(FS, value, dtype=np.float32) for _ in range(n)]


class NullIn(AudioInput):
    def start(self): ...
    def stop(self): ...
    def read(self, timeout=None): return None


class FakeOut(AudioOutput):
    def __init__(self):
        self.played, self.flushes, self.playing = [], 0, False
    def start(self): ...
    def stop(self): ...
    def play(self, audio):
        self.played.append(audio)
        self.playing = True
    def flush(self):
        self.flushes += 1
        self.playing = False
    def is_playing(self): return self.playing
    def pop_reference(self): return None
    async def wait_until_done(self):
        while self.playing:
            await asyncio.sleep(0.01)


class PassAEC(AECProcessor):
    available = True
    def process(self, mic, reference): return mic


class PassDenoise(Denoiser):
    def process(self, frame): return frame


class LevelVAD(VADDetector):
    def probability(self, frame): return 1.0 if frame.mean() > 0.05 else 0.0


class LevelVerifier(SpeakerVerifier):
    """Accepts audio whose voiced part looks like USER."""
    def __init__(self):
        self.calls = 0
    def embed(self, audio): return np.array([audio.mean()])
    def verify(self, audio, min_samples=None):
        self.calls += 1
        if audio.shape[0] < (min_samples if min_samples is not None else 6400):
            return VerificationResult(False, 0.0, 0.65)
        voiced = audio[audio > 0.05]
        score = 0.9 if voiced.size and abs(voiced.mean() - USER) < 0.05 else 0.2
        return VerificationResult(score >= 0.65, score, 0.65)


class FakeSTT(STTEngine):
    async def transcribe(self, audio): return "tell me a story"


class EndlessLLM(LLMClient):
    """Streams sentences forever until cancelled; records whether the stream was closed."""
    def __init__(self):
        self.closed = asyncio.Event()
        self.requests = []
    async def stream_chat(self, messages, **kw):
        self.requests.append(messages)
        try:
            i = 0
            while True:
                yield f"This is sentence number {i}. "
                i += 1
                await asyncio.sleep(0.02)
        finally:
            self.closed.set()


class FakeTTS(TTSEngine):
    async def synthesize(self, text):
        yield np.zeros(1600, dtype=np.float32)


@pytest.fixture
async def rig():
    cfg = Config()
    out, llm, ver = FakeOut(), EndlessLLM(), LevelVerifier()
    a = VoiceAssistant(cfg, audio_in=NullIn(), audio_out=out, aec=PassAEC(), denoiser=PassDenoise(), vad=LevelVAD(),
                       verifier=ver, stt=FakeSTT(), llm=llm, tts=FakeTTS())
    a._loop = asyncio.get_running_loop()
    a.sm.transition(State.LISTENING, "test")

    async def pump():
        while True:
            await a._handle(await a._events.get())

    pumper = asyncio.create_task(pump())
    yield a, out, llm, ver
    pumper.cancel()
    await a._cancel_response()


async def feed(a, fs):
    for f in fs:
        a.process_frame(f)
        await asyncio.sleep(0)


async def wait_for(pred, timeout=2.0):
    t0 = time.perf_counter()
    while not pred():
        if time.perf_counter() - t0 > timeout:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


async def test_other_speaker_is_ignored(rig):
    a, out, llm, _ = rig
    await feed(a, frames(OTHER, 40) + frames(0, 30))
    await asyncio.sleep(0.1)
    assert a.sm.state is State.LISTENING and not llm.requests and not out.played


async def test_user_utterance_triggers_response(rig):
    a, out, llm, _ = rig
    await feed(a, frames(USER, 40) + frames(0, 30))
    await wait_for(lambda: a.sm.state is State.SPEAKING)
    assert llm.requests[0][-1] == {"role": "user", "content": "tell me a story"}
    assert out.played


async def test_barge_in_by_user_stops_playback_and_cancels_llm(rig):
    a, out, llm, _ = rig
    await feed(a, frames(USER, 40) + frames(0, 30))
    await wait_for(lambda: a.sm.state is State.SPEAKING)
    flushes_before = out.flushes

    t0 = time.perf_counter()
    await feed(a, frames(USER, 14))  # 280 ms of the enrolled user talking over the assistant
    await wait_for(lambda: a.sm.state is State.LISTENING)
    stop_ms = (time.perf_counter() - t0) * 1000

    assert out.flushes > flushes_before
    await asyncio.wait_for(llm.closed.wait(), 0.1)          # SSE stream closed
    assert a._respond_task is None
    assert any(s[1] is State.INTERRUPTED for s in a.sm.history)
    assert stop_ms < 500
    assert a.history[-1]["content"].endswith("[interrupted by user]")


async def test_barge_in_needs_min_speech_duration(rig):
    a, out, llm, _ = rig
    await feed(a, frames(USER, 40) + frames(0, 30))
    await wait_for(lambda: a.sm.state is State.SPEAKING)
    await feed(a, frames(USER, 8) + frames(0, 30))  # 160 ms blip < 250 ms
    await asyncio.sleep(0.1)
    assert a.sm.state is State.SPEAKING and not llm.closed.is_set()


async def test_other_voice_cannot_barge_in(rig):
    a, out, llm, ver = rig
    await feed(a, frames(USER, 40) + frames(0, 30))
    await wait_for(lambda: a.sm.state is State.SPEAKING)
    calls = ver.calls
    await feed(a, frames(OTHER, 40))
    await asyncio.sleep(0.1)
    assert ver.calls > calls                    # it was checked...
    assert a.sm.state is State.SPEAKING          # ...and rejected
    assert not llm.closed.is_set()


class ShortLLM(EndlessLLM):
    async def stream_chat(self, messages, **kw):
        self.requests.append(messages)
        yield "Paris is the capital of France. "


async def test_interrupt_during_playback_after_llm_done_records_history_once(rig):
    a, out, _, _ = rig
    a.llm = ShortLLM()
    await feed(a, frames(USER, 40) + frames(0, 30))
    await wait_for(lambda: a.sm.state is State.SPEAKING and len(a.history) == 2)  # reply stored, still playing
    await feed(a, frames(USER, 14))
    await wait_for(lambda: a.sm.state is State.LISTENING)
    assert [m["role"] for m in a.history] == ["user", "assistant"]
    assert a.history[1]["content"] == "Paris is the capital of France. [interrupted by user]"


async def test_history_is_trimmed():
    cfg = Config()
    cfg.llm.history_turns = 2
    a = VoiceAssistant(cfg, audio_in=NullIn(), audio_out=FakeOut(), aec=PassAEC(), denoiser=PassDenoise(),
                       vad=LevelVAD(), verifier=LevelVerifier(), stt=FakeSTT(), llm=EndlessLLM(), tts=FakeTTS())
    for i in range(5):
        a._remember(f"u{i}", f"a{i}")
    assert [m["content"] for m in a.history] == ["u3", "a3", "u4", "a4"]


class DelayedLLM(EndlessLLM):
    """First token is slow, so the assistant sits in PROCESSING with nothing playing yet."""
    async def stream_chat(self, messages, **kw):
        self.requests.append(messages)
        try:
            await asyncio.sleep(0.5)
            while True:
                yield "Working on it. "
                await asyncio.sleep(0.02)
        finally:
            self.closed.set()


async def test_barge_in_during_processing_cancels_request():
    """Autonomous interruption also applies before playback starts: verified speech while the
    LLM is still working cancels the in-flight request and the new utterance is heard."""
    cfg = Config()
    llm = DelayedLLM()
    a = VoiceAssistant(cfg, audio_in=NullIn(), audio_out=FakeOut(), aec=PassAEC(), denoiser=PassDenoise(),
                       vad=LevelVAD(), verifier=LevelVerifier(), stt=FakeSTT(), llm=llm, tts=FakeTTS())
    a._loop = asyncio.get_running_loop()
    a.sm.transition(State.LISTENING, "test")
    pumper = asyncio.create_task(_pump(a))
    try:
        await feed(a, frames(USER, 40) + frames(0, 30))
        await wait_for(lambda: a.sm.state is State.PROCESSING)   # LLM slow -> stays here
        await feed(a, frames(USER, 14))                          # 280 ms verified user speech
        await wait_for(lambda: a.sm.state is State.LISTENING)
        await asyncio.wait_for(llm.closed.wait(), 1)             # in-flight request cancelled
        # the interrupting utterance is then processed as a normal request
        await feed(a, frames(0, 30))
        await wait_for(lambda: len(llm.requests) == 2)
    finally:
        pumper.cancel()
        await a._cancel_response()


async def _pump(a):
    while True:
        await a._handle(await a._events.get())


ECHO = 0.3  # residual TTS echo level


class MixVerifier(SpeakerVerifier):
    """Embedding = [share of USER-like samples, share of ECHO-like samples, share of anything
    else] over voiced frames, so a candidate dominated by echo matches the echo reference
    embedding rather than the enrolled profile."""
    profile = np.array([1.0, 0.0, 0.0])

    def embed(self, audio):
        voiced = np.abs(audio) > 0.05
        n = max(int(voiced.sum()), 1)
        near_u = (np.abs(audio - USER) < 0.05) & voiced
        near_e = (np.abs(audio - ECHO) < 0.05) & voiced
        return np.array([near_u.sum() / n, near_e.sum() / n, (voiced & ~near_u & ~near_e).sum() / n])

    def verify(self, audio, min_samples=None):
        if audio.shape[0] < (min_samples or 0):
            return VerificationResult(False, 0.0, 0.65)
        s = float(cosine_similarity(self.embed(audio), self.profile))
        return VerificationResult(s >= 0.65, s, 0.65)


class EchoOut(FakeOut):
    def pop_reference(self):
        return np.full(FS, ECHO, dtype=np.float32)


def _mix_rig():
    a = VoiceAssistant(Config(), audio_in=NullIn(), audio_out=EchoOut(), aec=PassAEC(),
                       denoiser=PassDenoise(), vad=LevelVAD(), verifier=MixVerifier(),
                       stt=FakeSTT(), llm=EndlessLLM(), tts=FakeTTS())
    a._loop = asyncio.get_running_loop()
    a.sm.transition(State.LISTENING, "t")
    a.sm.transition(State.PROCESSING, "t")
    a.sm.transition(State.SPEAKING, "t")
    a._ref_buf.append(np.full(FS, ECHO, dtype=np.float32))   # as if playback had been tapped
    return a


def _window(u, e, o):
    parts = [np.full(int(16000 * f), v, dtype=np.float32) for f, v in ((u, USER), (e, ECHO), (o, 0.45))]
    return np.concatenate(parts)


async def test_barge_relaxed_floor_accepts_user_over_echo():
    """User + residual echo: plain score 0.60 fails the 0.65 bar but beats the echo match -> accept."""
    a = _mix_rig()
    await a._check_barge_in(_window(0.35, 0.30, 0.35), {"speech_ms": 600})
    assert a.sm.state is State.LISTENING           # interrupted and back to listening
    assert any(s[1] is State.INTERRUPTED for s in a.sm.history)


async def test_barge_relaxed_floor_rejects_echo_dominant():
    """Mostly echo: score lands in the relaxed band but matches echo better -> rejected."""
    a = _mix_rig()
    await a._check_barge_in(_window(0.35, 0.45, 0.20), {"speech_ms": 600})
    assert a.sm.state is State.SPEAKING            # no interrupt
    assert not any(s[1] is State.INTERRUPTED for s in a.sm.history)


async def test_barge_accepts_when_near_end_survives_aec():
    """Same echo-tied embedding as the reject case, but the signal passes through the AEC
    nearly unattenuated (near_ratio 0.8): a real voice at the mic -> accept."""
    a = _mix_rig()
    await a._check_barge_in(_window(0.35, 0.45, 0.20), {"speech_ms": 600, "near_ratio": 0.8})
    assert a.sm.state is State.LISTENING
    assert any(s[1] is State.INTERRUPTED for s in a.sm.history)
