"""Gemini TTS streaming (SSE) and the streaming speech sink, plus planner per-attempt logging.
No network: httpx.MockTransport and fake LiveKit sessions."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from fdagent.core.actions import RequestReasoning, Speak
from fdagent.providers.gemini_reasoner import GeminiReasoner
from fdagent.providers.gemini_tts import (
    INTERACTIONS_URL,
    GeminiTTS,
    PcmAudio,
    TTSError,
    TTSStreamNotStarted,
    iter_sse_audio,
)
from fdagent.runtime.speech import SpeechChannel
from fdagent.voice.speech_sink import PrerenderedSpeechSink

from .fakes import until
from .test_reasoner_parsing import TOOLS_X

KEY = "test-placeholder-key"
PCM_A = b"\x01\x00" * 2400  # 0.1 s at 24 kHz
PCM_B = b"\x02\x00" * 7200  # 0.3 s


def ev_audio(pcm: bytes) -> dict:
    return {"event_type": "step.delta", "delta": {"type": "audio", "data": base64.b64encode(pcm).decode()}}


def sse(*events, raw_lines=()) -> bytes:
    out = [f"data: {json.dumps(e)}\n\n" for e in events] + [l + "\n" for l in raw_lines]
    return "".join(out).encode()


LIFECYCLE = [{"event_type": "interaction.created"}, {"event_type": "step.start"}]
END = [{"event_type": "step.stop"}, {"event_type": "interaction.completed"}]


async def lines_of(body: bytes):
    for l in body.decode().splitlines():
        yield l


def collect(agen):
    async def go():
        return [c async for c in agen]
    return asyncio.run(go())


def tts_with(responses, **kw):
    seen, sleeps = [], []
    it = iter(responses)

    def handler(req):
        seen.append(req)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    async def fake_sleep(s):
        sleeps.append(s)

    t = GeminiTTS(api_key=KEY, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=fake_sleep, **kw)
    return t, seen, sleeps


class FailingStream(httpx.AsyncByteStream):
    """Delivers some SSE bytes, then the connection breaks mid-body."""

    def __init__(self, first: bytes):
        self.first = first

    async def __aiter__(self):
        yield self.first
        raise httpx.ReadError("connection reset mid-stream")

    async def aclose(self):
        pass


# ── SSE parsing ─────────────────────────────────────────────────────────────
def test_sse_parsing_yields_only_audio_deltas_in_order():
    body = sse(*LIFECYCLE, ev_audio(PCM_A), {"event_type": "status_update"}, ev_audio(PCM_B), *END)
    assert collect(iter_sse_audio(lines_of(body))) == [PCM_A, PCM_B]


def test_sse_parsing_skips_empty_and_invalid_events():
    body = sse(ev_audio(PCM_A), raw_lines=[": keep-alive comment", "", "data:", "data: not-json", "data: [DONE]",
                                           "data: [1, 2]", 'data: {"delta": {"type": "audio"}}',
                                           'data: {"delta": {"type": "audio", "data": "***"}}', "event: step.delta"])
    assert collect(iter_sse_audio(lines_of(body))) == [PCM_A]


def test_sse_parsing_keeps_chunks_sample_aligned():
    odd1, odd2 = b"\x01\x00\x02", b"\x00\x03\x00"  # 3 bytes each; sample boundary crosses chunks
    out = collect(iter_sse_audio(lines_of(sse(ev_audio(odd1), ev_audio(odd2)))))
    assert all(len(c) % 2 == 0 for c in out) and b"".join(out) == odd1 + odd2


def test_sse_error_event_raises():
    with pytest.raises(TTSError, match="stream error event"):
        collect(iter_sse_audio(lines_of(sse(ev_audio(PCM_A), {"error": {"code": 500, "message": "boom"}}))))


# ── provider streaming ──────────────────────────────────────────────────────
def test_stream_yields_pcm_chunks_until_end_of_stream():
    t, seen, _ = tts_with([httpx.Response(200, content=sse(*LIFECYCLE, ev_audio(PCM_A), ev_audio(PCM_B), *END))])
    assert collect(t.stream("Hello there")) == [PCM_A, PCM_B]
    [r] = seen
    body = json.loads(r.content)
    assert str(r.url) == INTERACTIONS_URL and r.headers["x-goog-api-key"] == KEY
    assert body["stream"] is True and body["model"] == t.model and body["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    assert t.attempts == 1


def test_failure_before_first_chunk_is_retried():
    t, seen, sleeps = tts_with([httpx.Response(503), httpx.ConnectError("down"),
                                httpx.Response(200, content=sse(ev_audio(PCM_A), *END))])
    assert collect(t.stream("hi")) == [PCM_A]
    assert t.attempts == 3 and sleeps == [0.25, 0.5]


def test_stream_with_no_audio_is_retried_then_reports_not_started():
    t, _, _ = tts_with([httpx.Response(200, content=sse(*LIFECYCLE, *END))] * 2)
    with pytest.raises(TTSStreamNotStarted, match="stream ended without audio"):
        collect(t.stream("hi", attempts=2))
    assert t.attempts == 2


def test_non_retriable_error_reports_not_started_immediately():
    t, seen, sleeps = tts_with([httpx.Response(400, text="bad voice")])
    with pytest.raises(TTSStreamNotStarted, match="HTTP 400"):
        collect(t.stream("hi"))
    assert len(seen) == 1 and sleeps == []


def test_failure_after_first_chunk_is_never_restarted():
    t, seen, sleeps = tts_with([httpx.Response(200, stream=FailingStream(sse(ev_audio(PCM_A)))),
                                httpx.Response(200, content=sse(ev_audio(PCM_B)))])  # must never be requested

    async def go():
        got = []
        with pytest.raises(TTSError) as e:
            async for c in t.stream("hi"):
                got.append(c)
        return got, e.value

    got, err = asyncio.run(go())
    assert got == [PCM_A]
    assert not isinstance(err, TTSStreamNotStarted) and "after playback data began" in str(err)
    assert len(seen) == 1 and t.attempts == 1 and sleeps == []
    assert KEY not in str(err)


# ── streaming speech sink ───────────────────────────────────────────────────
class FakeHandle:
    def __init__(self, audio, seconds=0.0):
        self.audio, self.seconds, self.interrupted, self.frames, self.interrupts = audio, seconds, False, [], 0

    def interrupt(self, force=False):
        self.interrupts += 1
        self.interrupted = True

    async def wait_for_playout(self):
        async for f in self.audio:
            self.frames.append(f)
            await asyncio.sleep(self.seconds)


class FakeSession:
    def __init__(self, frame_seconds=0.0):
        self.handles, self.frame_seconds = [], frame_seconds

    def say(self, text, audio=None, allow_interruptions=True, add_to_chat_ctx=True):
        assert add_to_chat_ctx is False and audio is not None
        h = FakeHandle(audio, self.frame_seconds)
        self.handles.append((text, h))
        return h


class FakeStreamingTTS:
    def __init__(self, chunks=(PCM_A, PCM_B), fail_before=False, fail_after=0, delay=0.0):
        self.chunks, self.fail_before, self.fail_after, self.delay = chunks, fail_before, fail_after, delay
        self.stream_calls, self.synth_calls, self.closed = 0, 0, False

    async def stream(self, text):
        self.stream_calls += 1
        try:
            if self.fail_before:
                raise TTSStreamNotStarted("stream gave up after 2 attempts before any audio: HTTP 503")
            for i, c in enumerate(self.chunks):
                if self.fail_after and i == self.fail_after:
                    raise TTSError("audio stream interrupted after playback data began: ReadError")
                await asyncio.sleep(self.delay)
                yield c
        finally:
            self.closed = True

    async def synthesize(self, text):
        self.synth_calls += 1
        return PcmAudio(PCM_A + PCM_B, 24000, 1)


def run(coro):
    return asyncio.run(coro)


def samples(frames):
    return sum(f.samples_per_channel for f in frames)


def test_streaming_sink_plays_20ms_frames_incrementally():
    pytest.importorskip("livekit.rtc")
    notes = []
    session, tts = FakeSession(), FakeStreamingTTS()
    sink = PrerenderedSpeechSink(session, tts, observer=lambda **k: notes.append(k))
    assert run(sink.say("Hello")) is False
    [(text, h)] = session.handles
    assert text == "Hello" and samples(h.frames) == (len(PCM_A) + len(PCM_B)) // 2
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in h.frames)
    assert {f.samples_per_channel for f in h.frames} == {480}  # 20 ms frames; 0.4 s divides evenly
    assert tts.stream_calls == 1 and tts.synth_calls == 0 and tts.closed
    assert notes[0]["name"] == "tts_first_audio" and notes[0]["mode"] == "stream"


def test_prebuffer_waits_for_minimum_audio_before_first_frame():
    pytest.importorskip("livekit.rtc")
    order = []

    class Probe(FakeStreamingTTS):
        async def stream(self, text):
            for c in (b"\x00\x00" * 480, b"\x00\x00" * 480, b"\x00\x00" * 4800):  # 20 ms, 20 ms, 200 ms
                order.append(("chunk", len(c)))
                yield c

    class OrderHandle(FakeHandle):
        async def wait_for_playout(self):
            async for f in self.audio:
                order.append(("frame", f.samples_per_channel))

    session = FakeSession()
    session.say = lambda text, audio=None, **kw: session.handles.append(OrderHandle(audio)) or session.handles[-1]
    run(PrerenderedSpeechSink(session, Probe(), prebuffer_ms=200).say("x"))
    first_frame = order.index(("frame", 480))
    assert [o for o in order[:first_frame] if o[0] == "chunk"] == [("chunk", 960), ("chunk", 960), ("chunk", 9600)]


def test_stream_failure_before_audio_falls_back_to_unary_once():
    pytest.importorskip("livekit.rtc")
    notes = []
    session, tts = FakeSession(), FakeStreamingTTS(fail_before=True)
    run(PrerenderedSpeechSink(session, tts, observer=lambda **k: notes.append(k)).say("Hello"))
    assert tts.stream_calls == 1 and tts.synth_calls == 1 and len(session.handles) == 1
    assert [n["name"] for n in notes] == ["tts_stream_fallback", "tts_first_audio"]
    assert notes[1]["mode"] == "unary"


def test_stream_failure_after_audio_plays_delivered_audio_without_restart():
    pytest.importorskip("livekit.rtc")

    async def main():
        posted, marks = [], []
        session, tts = FakeSession(), FakeStreamingTTS(chunks=(PCM_A, PCM_B, PCM_B), fail_after=2)
        ch = SpeechChannel(PrerenderedSpeechSink(session, tts), posted.append, lambda **k: marks.append(k))
        ch.enqueue(Speak("Your order is on its way", "final", 1))
        await until(lambda: any(m.get("name") == "speech_end" for m in marks))
        await ch.aclose()
        return session, tts, marks

    session, tts, marks = run(main())
    [(_, h)] = session.handles  # one playout, never restarted
    assert samples(h.frames) == (len(PCM_A) + len(PCM_B)) // 2  # what arrived before the break still played
    assert tts.stream_calls == 1 and tts.synth_calls == 0
    failed = [m for m in marks if m["name"] == "speech_failed"]
    assert len(failed) == 1 and "after playback data began" in failed[0]["error"]


def test_barge_in_during_streamed_playback_interrupts_and_closes_stream():
    pytest.importorskip("livekit.rtc")

    async def main():
        posted, marks = [], []
        session = FakeSession(frame_seconds=0.02)
        tts = FakeStreamingTTS(chunks=(PCM_B,) * 20, delay=0.05)
        ch = SpeechChannel(PrerenderedSpeechSink(session, tts), posted.append, lambda **k: marks.append(k))
        ch.enqueue(Speak("A long streamed answer", "final", 1))
        await until(lambda: session.handles and session.handles[0][1].frames)
        ch.stop()
        await until(lambda: any(m.get("name") == "speech_end" for m in marks), timeout=2.0)
        await asyncio.sleep(0.1)
        await ch.aclose()
        return session, tts, marks

    session, tts, marks = run(main())
    [(_, h)] = session.handles
    assert h.interrupts == 1
    assert [m for m in marks if m["name"] == "speech_end"][0]["interrupted"] is True
    assert samples(h.frames) < 20 * len(PCM_B) // 2  # stopped early
    assert tts.stream_calls == 1 and tts.synth_calls == 0


def test_provider_output_is_compatible_with_the_sink():
    """Real GeminiTTS.stream (mock HTTP) feeding the real sink: PCM plays as 24 kHz mono frames."""
    pytest.importorskip("livekit.rtc")
    t, _, _ = tts_with([httpx.Response(200, content=sse(*LIFECYCLE, ev_audio(PCM_A), ev_audio(PCM_B), *END))])
    session = FakeSession()
    run(PrerenderedSpeechSink(session, t).say("Hello"))
    [(_, h)] = session.handles
    assert samples(h.frames) == (len(PCM_A) + len(PCM_B)) // 2 and h.frames[0].sample_rate == 24000


# ── planner per-attempt logging ─────────────────────────────────────────────
def test_planner_attempts_are_logged_with_retry_count_and_status():
    ok = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
          "choices": [{"index": 0, "finish_reason": "stop",
                       "message": {"role": "assistant", "content": json.dumps({"keep": [], "new_calls": [], "reply": "Hi"})}}]}
    responses = iter([httpx.Response(503, headers={"retry-after": "0"}, json={"error": {"message": "busy"}}),
                      httpx.Response(200, json=ok)])
    attempts = []
    r = GeminiReasoner(TOOLS_X, api_key="test-placeholder", max_retries=1, on_attempt=attempts.append,
                       transport=httpx.MockTransport(lambda req: next(responses)))
    snap = {"generation": 1, "conversation": [], "calls": [], "orphan_effects": [], "transcript": []}
    draft = run(r.propose(RequestReasoning("req-7", 1, snap)))
    assert draft.reply == "Hi"
    assert [(a["request_id"], a["retry_count"], a["status"], a["retry_after"]) for a in attempts] == [
        ("req-7", "0", 503, "0"), ("req-7", "1", 200, None)]
    assert all(isinstance(a["attempt_s"], float) for a in attempts)
    assert not any("test-placeholder" in json.dumps(a) for a in attempts)
