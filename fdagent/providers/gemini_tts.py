"""Gemini TTS (free tier, no billing account) over the documented REST ``interactions`` API.

Returns 16-bit PCM; the LiveKit sink plays it with ``session.say(text, audio=...)``, so no
TTS plugin (and no Google Cloud dependency) is needed. Transient failures (429, 5xx,
timeouts, transport errors) are retried with exponential backoff; other HTTP errors fail
immediately. Errors never include the API key.
"""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from .gemini_reasoner import gemini_api_key

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-lite-tts"  # stable; free tier per ai.google.dev pricing
DEFAULT_GEMINI_VOICE = "Kore"
RAW_PCM_RATE = 24000  # documented format for headerless audio/l16 output
_RETRIABLE_STATUS = {429, 500, 502, 503, 504}


class TTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class PcmAudio:
    data: bytes  # little-endian int16
    sample_rate: int
    channels: int = 1


def decode_audio_response(obj: dict[str, Any]) -> PcmAudio:
    datas = [
        c["data"]
        for step in obj.get("steps", [])
        if step.get("type") == "model_output"
        for c in step.get("content", [])
        if c.get("type") == "audio" and c.get("data")
    ]
    if not datas:
        raise TTSError("response contained no audio")
    raw = base64.b64decode(datas[-1])
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw)) as w:
            if w.getsampwidth() != 2:
                raise TTSError(f"unsupported sample width {w.getsampwidth()}")
            return PcmAudio(w.readframes(w.getnframes()), w.getframerate(), w.getnchannels())
    return PcmAudio(raw, RAW_PCM_RATE, 1)


class GeminiTTS:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_TTS_MODEL,
        voice: str = DEFAULT_GEMINI_VOICE,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 3,
        backoff_s: float = 0.25,
        timeout_s: float = 15.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self.voice = voice
        self._api_key = api_key
        self._client = client
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self._sleep = sleep
        self.attempts = 0  # total HTTP attempts (observability / tests)

    def _body(self, text: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": self.voice}]},
        }

    async def synthesize(self, text: str) -> PcmAudio:
        key = self._api_key or gemini_api_key()
        if not key:
            raise TTSError("GOOGLE_API_KEY is not set (needed for Gemini TTS)")
        if self._client is None:
            self._client = httpx.AsyncClient()
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        last = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            self.attempts += 1
            try:
                r = await self._client.post(INTERACTIONS_URL, json=self._body(text), headers=headers,
                                            timeout=self.timeout_s)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"{type(e).__name__}"
            else:
                if r.status_code == 200:
                    return decode_audio_response(r.json())
                if r.status_code not in _RETRIABLE_STATUS:
                    raise TTSError(f"HTTP {r.status_code}: {r.text[:200]}")
                last = f"HTTP {r.status_code}"
            if attempt < self.max_attempts:
                await self._sleep(self.backoff_s * 2 ** (attempt - 1))
        raise TTSError(f"gave up after {self.max_attempts} attempts: {last}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
