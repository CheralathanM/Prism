"""Local Whisper STT adapter: audio conversion, LiveKit STT contract, lazy/threaded inference.
Uses a fake Whisper pipeline, so no model weights or transformers are needed to run it."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

rtc = pytest.importorskip("livekit.rtc")
from livekit.agents import stt  # noqa: E402

from fdagent.voice.local_whisper import (  # noqa: E402
    DEFAULT_WHISPER_MODEL,
    LocalWhisperSTT,
    WhisperTranscriber,
    frames_to_float32_16k,
)


def tone(seconds: float, rate: int, channels: int = 1, amp: int = 8000) -> rtc.AudioFrame:
    n = int(seconds * rate)
    mono = (amp * np.sin(2 * np.pi * 220 * np.arange(n) / rate)).astype(np.int16)
    data = np.repeat(mono, channels) if channels > 1 else mono
    return rtc.AudioFrame(data.tobytes(), rate, channels, n)


class FakePipe:
    def __init__(self, text="rooms in Eastvale"):
        self.text, self.calls, self.threads = text, [], []

    def __call__(self, inputs, **kw):
        self.calls.append((inputs, kw))
        self.threads.append(threading.get_ident())
        return {"text": f"  {self.text} "}


def test_default_model_is_tiny_en_and_agent_default_matches():
    from fdagent.voice import livekit_agent

    assert DEFAULT_WHISPER_MODEL == "openai/whisper-tiny.en"
    assert livekit_agent.DEFAULT_WHISPER_MODEL == DEFAULT_WHISPER_MODEL
    assert livekit_agent.AgentSettings.from_env({"FDAGENT_WHISPER_MODEL": "openai/whisper-base.en"}).whisper_model \
        == "openai/whisper-base.en"


def test_conversion_resamples_to_16k_mono_float32():
    for rate, channels in ((48000, 1), (24000, 1), (48000, 2), (16000, 1)):
        out = frames_to_float32_16k(tone(0.5, rate, channels))
        assert out.dtype == np.float32 and abs(out.size - 8000) <= 160, (rate, channels, out.size)
        assert 0.1 < np.abs(out).max() <= 1.0
    merged = frames_to_float32_16k([tone(0.25, 48000), tone(0.25, 48000)])
    assert abs(merged.size - 8000) <= 160
    assert frames_to_float32_16k([]).size == 0


def test_transcriber_is_lazy_and_loaded_once():
    made = []
    pipe = FakePipe()
    t = WhisperTranscriber("openai/whisper-small.en", pipeline_factory=lambda m, d: made.append((m, d)) or pipe)
    assert made == []  # constructing never loads weights
    assert t.transcribe(np.zeros(0, dtype=np.float32)) == ""  # empty audio: no load either
    assert made == []
    assert t.transcribe(np.ones(1600, dtype=np.float32) * 0.1) == "rooms in Eastvale"
    t.transcribe(np.ones(1600, dtype=np.float32) * 0.1)
    assert made == [("openai/whisper-small.en", "cpu")]
    inputs, kw = pipe.calls[0]
    assert inputs["sampling_rate"] == 16000 and kw == {}  # English-only model: no language forcing


def test_multilingual_checkpoint_is_forced_to_english():
    pipe = FakePipe()
    WhisperTranscriber("openai/whisper-small", pipeline_factory=lambda m, d: pipe).transcribe(np.ones(160, np.float32))
    assert pipe.calls[0][1] == {"generate_kwargs": {"language": "english", "task": "transcribe"}}


def test_livekit_stt_contract_returns_final_transcript_off_the_loop():
    pipe = FakePipe()
    s = LocalWhisperSTT(WhisperTranscriber(DEFAULT_WHISPER_MODEL, pipeline_factory=lambda m, d: pipe))
    assert s.capabilities.streaming is False and s.capabilities.interim_results is False

    async def main():
        return await s.recognize([tone(0.3, 48000), tone(0.3, 48000)]), threading.get_ident()

    ev, loop_thread = asyncio.run(main())
    assert ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert ev.alternatives[0].text == "rooms in Eastvale" and ev.alternatives[0].language == "en"
    audio = pipe.calls[0][0]["raw"]
    assert audio.dtype == np.float32 and abs(audio.size - 9600) <= 200
    assert pipe.threads[0] != loop_thread  # inference ran in a worker thread
    assert s.provider == "local-whisper" and s.model == DEFAULT_WHISPER_MODEL
