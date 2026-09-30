"""Local Piper TTS backend (default benchmark TTS): format, streaming, failures, sink compatibility.
Uses a fake voice with Piper's chunk API, so no model or piper package is needed on Windows."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from fdagent.core.actions import Speak
from fdagent.providers.local_piper_tts import PiperTTS, load_piper_voice
from fdagent.providers.tts_base import PcmAudio, TTSError, TTSStreamNotStarted
from fdagent.runtime.speech import SpeechChannel
from fdagent.voice.speech_sink import PrerenderedSpeechSink

from .fakes import until

S1 = b"\x01\x00" * 4410  # 0.2 s at 22.05 kHz
S2 = b"\x02\x00" * 8820  # 0.4 s


class FakeVoice:
    """Mimics piper.PiperVoice: .config.sample_rate and .synthesize(text) -> AudioChunk iterator."""

    def __init__(self, sentences=(S1, S2), fail_at=None, delay=0.0, channels=1, width=2):
        self.config = SimpleNamespace(sample_rate=22050)
        self.sentences, self.fail_at, self.delay = sentences, fail_at, delay
        self.channels, self.width = channels, width
        self.calls, self.produced, self.threads = 0, 0, set()

    def synthesize(self, text):
        self.calls += 1
        self.threads.add(threading.get_ident())
        for i, pcm in enumerate(self.sentences):
            if self.fail_at == i:
                raise RuntimeError("onnx inference failed")
            if self.delay:
                import time
                time.sleep(self.delay)
            self.produced += 1
            yield SimpleNamespace(audio_int16_bytes=pcm, sample_channels=self.channels, sample_width=self.width)


def run(coro):
    return asyncio.run(coro)


def collect(agen):
    async def go():
        return [c async for c in agen]
    return run(go())


def test_synthesize_returns_mono_pcm16_at_voice_rate_off_the_event_loop():
    voice = FakeVoice()
    audio = run(PiperTTS(voice=voice).synthesize("Hello there."))
    assert audio == PcmAudio(S1 + S2, 22050, 1)
    assert threading.get_ident() not in voice.threads  # synthesis ran in a worker thread


def test_stream_yields_one_chunk_per_sentence_in_order():
    tts = PiperTTS(voice=FakeVoice())
    assert collect(tts.stream("One. Two.")) == [S1, S2]
    assert tts.stream_sample_rate == 22050


def test_stream_failure_before_audio_is_not_started():
    with pytest.raises(TTSStreamNotStarted, match="onnx inference failed"):
        collect(PiperTTS(voice=FakeVoice(fail_at=0)).stream("x"))
    with pytest.raises(TTSStreamNotStarted, match="no audio"):
        collect(PiperTTS(voice=FakeVoice(sentences=(b"",))).stream("x"))


def test_stream_failure_after_audio_is_a_tts_error_not_a_restart():
    voice = FakeVoice(sentences=(S1, S2, S2), fail_at=1)

    async def go():
        got = []
        with pytest.raises(TTSError) as e:
            async for c in PiperTTS(voice=voice).stream("x"):
                got.append(c)
        return got, e.value

    got, err = run(go())
    assert got == [S1] and not isinstance(err, TTSStreamNotStarted) and voice.calls == 1


def test_unsupported_format_is_rejected():
    with pytest.raises(TTSError, match="unsupported Piper format"):
        run(PiperTTS(voice=FakeVoice(channels=2)).synthesize("x"))


def test_missing_voice_file_fails_loudly(tmp_path):
    with pytest.raises(TTSError, match="fetch_piper_voice"):
        load_piper_voice(str(tmp_path / "nope.onnx"))
    with pytest.raises(TTSError, match="fetch_piper_voice"):
        run(PiperTTS(model_path=str(tmp_path / "nope.onnx")).synthesize("x"))


def test_voice_is_loaded_once_lazily():
    loads = []
    tts = PiperTTS(model_path="v.onnx", loader=lambda p: loads.append(p) or FakeVoice())
    assert loads == []
    run(tts.synthesize("a"))
    run(tts.synthesize("b"))
    assert loads == ["v.onnx"]


# ── sink compatibility / fallback ───────────────────────────────────────────
class FakeHandle:
    def __init__(self, audio, per_frame_s=0.0):
        self.audio, self.per_frame_s, self.interrupted, self.frames, self.interrupts = audio, per_frame_s, False, [], 0

    def interrupt(self, force=False):
        self.interrupts += 1
        self.interrupted = True

    async def wait_for_playout(self):
        async for f in self.audio:
            self.frames.append(f)
            await asyncio.sleep(self.per_frame_s)


class FakeSession:
    def __init__(self, per_frame_s=0.0):
        self.handles, self.per_frame_s = [], per_frame_s

    def say(self, text, audio=None, allow_interruptions=True, add_to_chat_ctx=True):
        h = FakeHandle(audio, self.per_frame_s)
        self.handles.append(h)
        return h


def test_piper_output_plays_through_the_sink_at_its_own_sample_rate():
    pytest.importorskip("livekit.rtc")
    session = FakeSession()
    run(PrerenderedSpeechSink(session, PiperTTS(voice=FakeVoice())).say("Hello. World."))
    [h] = session.handles
    assert {f.sample_rate for f in h.frames} == {22050}
    assert {f.samples_per_channel for f in h.frames} == {441}  # 20 ms at 22.05 kHz
    assert sum(f.samples_per_channel for f in h.frames) == (len(S1) + len(S2)) // 2


def test_stream_that_never_starts_falls_back_to_unary_synthesis():
    pytest.importorskip("livekit.rtc")

    class FlakyStream(PiperTTS):
        async def stream(self, text):
            raise TTSStreamNotStarted("stream unavailable")
            yield b""  # pragma: no cover

    notes, session = [], FakeSession()
    run(PrerenderedSpeechSink(session, FlakyStream(voice=FakeVoice()), observer=lambda **k: notes.append(k)).say("Hi."))
    assert [n["name"] for n in notes] == ["tts_stream_fallback", "tts_first_audio"] and notes[1]["mode"] == "unary"
    assert sum(f.samples_per_channel for f in session.handles[0].frames) == (len(S1) + len(S2)) // 2


def test_barge_in_stops_piper_before_remaining_sentences_are_synthesized():
    pytest.importorskip("livekit.rtc")
    voice = FakeVoice(sentences=(S1,) * 30, delay=0.02)

    async def main():
        marks, session = [], FakeSession(per_frame_s=0.02)
        ch = SpeechChannel(PrerenderedSpeechSink(session, PiperTTS(voice=voice)), lambda e: None,
                           lambda **k: marks.append(k))
        ch.enqueue(Speak("A long local answer.", "final", 1))
        await until(lambda: session.handles and session.handles[0].frames)
        ch.stop()
        await until(lambda: any(m.get("name") == "speech_end" for m in marks), timeout=2.0)
        await asyncio.sleep(0.3)  # give the worker thread time to observe the stop flag
        await ch.aclose()
        return session, marks

    session, marks = run(main())
    assert session.handles[0].interrupts == 1
    assert [m for m in marks if m["name"] == "speech_end"][0]["interrupted"] is True
    assert voice.produced < 30  # synthesis stopped early instead of rendering the whole reply
