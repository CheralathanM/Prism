"""Gemini TTS adapter (request shape, decoding, retry/failure) and the pre-rendered speech sink.
HTTP is faked with httpx.MockTransport; no network."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

import httpx
import pytest

from fdagent.core.actions import Speak
from fdagent.core.events import AgentSpeechEnded
from fdagent.providers.gemini_tts import (
    DEFAULT_GEMINI_TTS_MODEL,
    INTERACTIONS_URL,
    GeminiTTS,
    PcmAudio,
    TTSError,
    decode_audio_response,
)
from fdagent.runtime.speech import SpeechChannel
from fdagent.voice.speech_sink import PrerenderedSpeechSink

from .fakes import until

KEY = "test-placeholder-key"
PCM = (b"\x01\x00\xff\xff" * 2400)  # 4800 int16 samples = 0.2 s at 24 kHz


def wav_bytes(pcm=PCM, rate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def audio_response(raw: bytes) -> dict:
    return {"steps": [{"type": "thought", "content": []},
                      {"type": "model_output", "content": [{"type": "audio", "data": base64.b64encode(raw).decode()}]}]}


def tts_with(responses, **kw):
    """responses: list of httpx.Response | Exception, consumed in order."""
    seen, sleeps = [], []
    it = iter(responses)

    def handler(request: httpx.Request):
        seen.append(request)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    async def fake_sleep(s):
        sleeps.append(s)

    tts = GeminiTTS(api_key=KEY, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=fake_sleep, **kw)
    return tts, seen, sleeps


def run(coro):
    return asyncio.run(coro)


def test_request_shape_and_wav_decoding():
    tts, seen, _ = tts_with([httpx.Response(200, json=audio_response(wav_bytes()))])
    audio = run(tts.synthesize("Hello there"))
    assert audio == PcmAudio(PCM, 24000, 1)
    [r] = seen
    assert str(r.url) == INTERACTIONS_URL and r.headers["x-goog-api-key"] == KEY
    body = json.loads(r.content)
    assert body == {"model": DEFAULT_GEMINI_TTS_MODEL,
                    "input": [{"type": "user_input", "content": [{"type": "text", "text": "Hello there"}]}],
                    "response_format": {"type": "audio"},
                    "generation_config": {"speech_config": [{"voice": "Kore"}]}}


def test_headerless_pcm_is_24k_mono():
    assert decode_audio_response(audio_response(PCM)) == PcmAudio(PCM, 24000, 1)


def test_transient_failures_are_retried_with_backoff():
    tts, seen, sleeps = tts_with([httpx.Response(503), httpx.ConnectError("boom"),
                                  httpx.Response(200, json=audio_response(wav_bytes()))])
    assert run(tts.synthesize("hi")).data == PCM
    assert len(seen) == 3 and tts.attempts == 3
    assert sleeps == [0.25, 0.5]


def test_rate_limit_exhaustion_raises_without_leaking_key():
    tts, _, sleeps = tts_with([httpx.Response(429)] * 3)
    with pytest.raises(TTSError) as e:
        run(tts.synthesize("hi"))
    assert "gave up after 3 attempts: HTTP 429" in str(e.value) and KEY not in str(e.value)
    assert sleeps == [0.25, 0.5]


def test_non_retriable_error_fails_immediately():
    tts, seen, sleeps = tts_with([httpx.Response(400, text="bad voice")])
    with pytest.raises(TTSError, match="HTTP 400"):
        run(tts.synthesize("hi"))
    assert len(seen) == 1 and sleeps == []


def test_empty_audio_and_missing_key(monkeypatch):
    with pytest.raises(TTSError, match="no audio"):
        decode_audio_response({"steps": [{"type": "model_output", "content": [{"type": "text", "text": "x"}]}]})
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(TTSError, match="GOOGLE_API_KEY"):
        run(GeminiTTS().synthesize("hi"))


# ── pre-rendered sink + speech channel behaviour ───────────────────────────
class FakeHandle:
    def __init__(self, audio, seconds=0.01):
        self.audio, self.seconds, self.interrupted, self.frames = audio, seconds, False, []

    def interrupt(self, force=False):
        self.interrupted = True

    async def wait_for_playout(self):
        async for f in self.audio:
            self.frames.append(f)
        await asyncio.sleep(self.seconds)


class FakeSession:
    def __init__(self):
        self.handles = []

    def say(self, text, audio=None, allow_interruptions=True, add_to_chat_ctx=True):
        assert add_to_chat_ctx is False and audio is not None
        h = FakeHandle(audio)
        self.handles.append((text, h))
        return h


class ScriptedTTS:
    def __init__(self, fail_texts=(), delay=0.0):
        self.fail_texts, self.delay, self.calls = set(fail_texts), delay, []

    async def synthesize(self, text):
        self.calls.append(text)
        await asyncio.sleep(self.delay)
        if text in self.fail_texts:
            raise TTSError("gave up after 3 attempts: HTTP 503")
        return PcmAudio(PCM, 24000, 1)


def test_sink_streams_20ms_frames_through_livekit():
    pytest.importorskip("livekit.rtc")
    session = FakeSession()
    interrupted = run(PrerenderedSpeechSink(session, ScriptedTTS()).say("Hello"))
    [(text, h)] = session.handles
    assert text == "Hello" and interrupted is False
    assert len(h.frames) == 10 and all(f.sample_rate == 24000 and f.samples_per_channel == 480 for f in h.frames)


def test_tts_failure_is_reported_and_the_channel_keeps_working():
    pytest.importorskip("livekit.rtc")

    async def main():
        posted, marks = [], []
        session = FakeSession()
        ch = SpeechChannel(PrerenderedSpeechSink(session, ScriptedTTS(fail_texts={"first"})), posted.append,
                           lambda **k: marks.append(k))
        ch.enqueue(Speak("first", "final", 1))
        ch.enqueue(Speak("second", "final", 1))
        await until(lambda: [m for m in marks if m.get("name") == "speech_end"][1:])
        await ch.aclose()
        return posted, marks, session

    posted, marks, session = run(main())
    failed = [m for m in marks if m["name"] == "speech_failed"]
    assert len(failed) == 1 and failed[0]["speech_kind"] == "final" and "HTTP 503" in failed[0]["error"]
    assert [t for t, _ in session.handles] == ["second"]  # failure did not stop later speech
    assert sum(isinstance(e, AgentSpeechEnded) for e in posted) == 2  # kernel never left 'speaking'


def test_barge_in_during_synthesis_cancels_before_playout():
    async def main():
        posted, marks = [], []
        session = FakeSession()
        ch = SpeechChannel(PrerenderedSpeechSink(session, ScriptedTTS(delay=5.0)), posted.append,
                           lambda **k: marks.append(k))
        ch.enqueue(Speak("long answer", "final", 1))
        await until(lambda: any(m.get("name") == "speech_start" for m in marks))
        ch.stop()
        await until(lambda: any(m.get("name") == "speech_end" for m in marks), timeout=1.0)
        await ch.aclose()
        return posted, marks, session

    posted, marks, session = run(main())
    assert session.handles == []  # nothing reached LiveKit
    assert [m for m in marks if m["name"] == "speech_end"][0]["interrupted"] is True
