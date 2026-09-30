"""Local Whisper STT for LiveKit (runs entirely on this machine; no API calls).

Uses OpenAI's open Whisper checkpoints through Hugging Face ``transformers`` (already
installed as a NeMo dependency; weights are downloaded once from the Hugging Face hub).
Non-streaming: Silero VAD segments the audio and ``AgentSession`` hands each segment to
``_recognize_impl``, exactly like the template's non-streaming ``whisper-1`` STT.
Inference runs in a worker thread so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

# base.en: on the local synthetic regression set (numbers, IDs, dates, names; clean and Opus
# round-trip) it heard 9-10/10 key values vs 8-9/10 for tiny.en, fixing spoken-ID errors, at ~2.7 s
# for a 6 s turn with a 4-thread cap on this CPU (tiny.en ~1.5 s). Override: FDAGENT_WHISPER_MODEL.
DEFAULT_WHISPER_MODEL = "openai/whisper-base.en"
TARGET_RATE = 16000

PipelineFactory = Callable[[str, str], Any]


def _transformers_pipeline(model_id: str, device: str) -> Any:
    from transformers import pipeline  # heavy import, only when actually loading

    return pipeline("automatic-speech-recognition", model=model_id, device=device)


class WhisperTranscriber:
    """Thread-safe, lazily loaded Whisper. ``transcribe`` takes float32 mono 16 kHz audio."""

    def __init__(self, model_id: str = DEFAULT_WHISPER_MODEL, device: str = "cpu",
                 pipeline_factory: PipelineFactory | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._factory = pipeline_factory or _transformers_pipeline
        self._pipe: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        with self._lock:
            if self._pipe is None:
                self._pipe = self._factory(self.model_id, self.device)
            return self._pipe

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        pipe = self.load()
        kwargs: dict[str, Any] = {}
        if not self.model_id.endswith(".en"):
            kwargs["generate_kwargs"] = {"language": "english", "task": "transcribe"}
        with self._lock:  # one inference at a time per model instance
            out = pipe({"raw": audio, "sampling_rate": TARGET_RATE}, **kwargs)
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        return text.strip()


_CACHE: dict[tuple[str, str], WhisperTranscriber] = {}


def get_transcriber(model_id: str = DEFAULT_WHISPER_MODEL, device: str = "cpu") -> WhisperTranscriber:
    """Process-wide cache: load the model once per worker process, not once per room."""
    key = (model_id, device)
    if key not in _CACHE:
        _CACHE[key] = WhisperTranscriber(model_id, device)
    return _CACHE[key]


def frames_to_float32_16k(buffer: Any) -> np.ndarray:
    """LiveKit AudioBuffer (frame or list of frames, int16, any rate/channels) → float32 16 kHz mono."""
    if isinstance(buffer, list):
        if not buffer:
            return np.zeros(0, dtype=np.float32)
        frame = utils.merge_frames(buffer)
    else:
        frame = buffer
    pcm = np.frombuffer(bytes(frame.data), dtype=np.int16)
    if pcm.size == 0:
        return np.zeros(0, dtype=np.float32)
    if frame.num_channels > 1:
        pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
    if frame.sample_rate != TARGET_RATE:
        rs = rtc.AudioResampler(frame.sample_rate, TARGET_RATE, num_channels=1)
        mono = rtc.AudioFrame(pcm.tobytes(), frame.sample_rate, 1, len(pcm))
        out = rs.push(mono) + rs.flush()
        pcm = np.concatenate([np.frombuffer(bytes(f.data), dtype=np.int16) for f in out]) if out else pcm[:0]
    return pcm.astype(np.float32) / 32768.0


class LocalWhisperSTT(stt.STT):
    def __init__(self, transcriber: WhisperTranscriber | None = None, model_id: str = DEFAULT_WHISPER_MODEL) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._transcriber = transcriber or get_transcriber(model_id)

    @property
    def model(self) -> str:
        return self._transcriber.model_id

    @property
    def provider(self) -> str:
        return "local-whisper"

    def prewarm(self) -> None:
        """Load weights ahead of the first utterance (call from a thread)."""
        self._transcriber.load()

    async def _recognize_impl(
        self,
        buffer: Any,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        audio = frames_to_float32_16k(buffer)
        text = await asyncio.to_thread(self._transcriber.transcribe, audio)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text=text)],
        )
