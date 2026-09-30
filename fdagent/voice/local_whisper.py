"""Local Whisper STT for LiveKit (runs entirely on this machine; no API calls).

Uses OpenAI's open Whisper checkpoints through Hugging Face ``transformers``. Non-streaming:
Silero VAD segments the audio and ``AgentSession`` hands each segment to ``_recognize_impl``,
like the template's non-streaming ``whisper-1`` STT.

In the agent, decoding runs in a dedicated STT child process (``stt_process.SttProcessClient``)
so it can never hold the agent's GIL; this adapter only converts audio and awaits the result.
Each segment's lifecycle is reported to an optional ``observer`` (the ingress bridge), which
lets the kernel treat a segment as unresolved while it is still being transcribed:
``observer("started")`` then either ``observer("final", text=...)`` or
``observer("failed", error=...)``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

from fdagent.voice.whisper_core import (  # noqa: F401  (re-exported for callers/tests)
    DEFAULT_WHISPER_MODEL,
    TARGET_RATE,
    WhisperTranscriber,
    _transformers_pipeline,
    get_transcriber,
)


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
    def __init__(self, transcriber: Any = None, model_id: str = DEFAULT_WHISPER_MODEL,
                 observer: Callable[..., None] | None = None) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._transcriber = transcriber or get_transcriber(model_id)
        self.observer = observer

    @property
    def model(self) -> str:
        return self._transcriber.model_id

    @property
    def provider(self) -> str:
        return "local-whisper"

    def prewarm(self) -> None:
        """Load weights ahead of the first utterance (call from a thread)."""
        self._transcriber.load()

    def _notify(self, kind: str, **fields: Any) -> None:
        if self.observer is not None:
            self.observer(kind, **fields)

    async def _transcribe(self, audio: np.ndarray) -> str:
        if hasattr(self._transcriber, "atranscribe"):  # dedicated STT process (agent path)
            return await self._transcriber.atranscribe(audio)
        return await asyncio.to_thread(self._transcriber.transcribe, audio)  # in-process (tests/tools)

    async def _recognize_impl(
        self,
        buffer: Any,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        audio = frames_to_float32_16k(buffer)
        self._notify("started")
        try:
            text = await self._transcribe(audio)
        except asyncio.CancelledError:
            self._notify("failed", error="cancelled")
            raise
        except Exception as e:  # noqa: BLE001 - explicit failure; do not let LiveKit retry blindly
            self._notify("failed", error=f"{type(e).__name__}: {e}"[:300])
            text = ""
        else:
            self._notify("final", text=text)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language="en", text=text)],
        )
