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
import json
import wave
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from .gemini_reasoner import gemini_api_key
from .tts_base import PcmAudio, TTSError, TTSStreamNotStarted

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-lite-tts"  # stable; free tier per ai.google.dev pricing
DEFAULT_GEMINI_VOICE = "Kore"
RAW_PCM_RATE = 24000  # documented format for headerless audio/l16 output
_RETRIABLE_STATUS = {429, 500, 502, 503, 504}


__all__ = ["GeminiTTS", "PcmAudio", "TTSError", "TTSStreamNotStarted", "decode_audio_response", "iter_sse_audio"]


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


async def iter_sse_audio(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Streaming TTS events → PCM16 byte chunks (24 kHz mono, per the documented audio/l16 format).

    Only ``data:`` lines carrying ``delta.type == "audio"`` produce output; other events,
    blank/comment lines and undecodable JSON are skipped. An ``error`` event raises. A trailing
    odd byte is carried to the next chunk so every yielded chunk is sample-aligned.
    """
    carry = b""
    async for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("error"):
            raise TTSError(f"stream error event: {str(ev['error'])[:200]}")
        delta = ev.get("delta")
        if not isinstance(delta, dict) or delta.get("type") != "audio" or not delta.get("data"):
            continue
        try:
            data = carry + base64.b64decode(delta["data"])
        except (ValueError, TypeError):
            continue
        cut = len(data) - (len(data) % 2)
        data, carry = data[:cut], data[cut:]
        if data:
            yield data


class GeminiTTS:
    stream_sample_rate = RAW_PCM_RATE  # streamed audio/l16 chunks

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

    async def stream(self, text: str, attempts: int | None = None) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks (``RAW_PCM_RATE`` Hz, mono) as they are synthesized.

        Retries transient failures only while no audio has been yielded. Once playback data
        has been produced, any failure raises ``TTSError`` and is never retried (an utterance
        is never restarted). If every attempt fails before the first chunk, raises
        ``TTSStreamNotStarted`` so the caller can fall back to ``synthesize``.
        """
        key = self._api_key or gemini_api_key()
        if not key:
            raise TTSStreamNotStarted("GOOGLE_API_KEY is not set (needed for Gemini TTS)")
        if self._client is None:
            self._client = httpx.AsyncClient()
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        body = {**self._body(text), "stream": True}
        attempts = attempts or self.max_attempts
        last = "no attempt made"
        for attempt in range(1, attempts + 1):
            self.attempts += 1
            started = False
            try:
                async with self._client.stream("POST", INTERACTIONS_URL, json=body, headers=headers,
                                               timeout=self.timeout_s) as r:
                    if r.status_code != 200:
                        detail = (await r.aread())[:200].decode(errors="replace")
                        if r.status_code not in _RETRIABLE_STATUS:
                            raise TTSStreamNotStarted(f"HTTP {r.status_code}: {detail}")
                        last = f"HTTP {r.status_code}"
                    else:
                        async for chunk in iter_sse_audio(r.aiter_lines()):
                            started = True
                            yield chunk
                        if started:
                            return
                        last = "stream ended without audio"
            except (httpx.TimeoutException, httpx.TransportError, TTSError) as e:
                if started:
                    raise TTSError(f"audio stream interrupted after playback data began: {type(e).__name__}") from None
                if isinstance(e, TTSStreamNotStarted):
                    raise
                last = f"{type(e).__name__}"
            if attempt < attempts:
                await self._sleep(self.backoff_s * 2 ** (attempt - 1))
        raise TTSStreamNotStarted(f"stream gave up after {attempts} attempts before any audio: {last}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
