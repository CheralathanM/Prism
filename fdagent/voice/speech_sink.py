"""Protocol adapter · egress (speech): kernel Speak actions → LiveKit TTS playout.

Duck-typed against ``AgentSession.say`` / ``SpeechHandle`` so it can be tested with fakes.
Speech is not added to LiveKit's chat context: the kernel owns conversation state.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable


class LiveKitSpeechSink:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def say(self, text: str) -> bool:
        handle = self._session.say(text, allow_interruptions=True, add_to_chat_ctx=False)
        try:
            await handle.wait_for_playout()
        except asyncio.CancelledError:
            handle.interrupt()  # kernel barge-in (StopSpeaking) or shutdown
            raise
        # LiveKit may interrupt playout itself when the user starts speaking.
        return bool(handle.interrupted)


class PrerenderedSpeechSink:
    """Play externally synthesized speech (e.g. Gemini TTS) through LiveKit.

    ``session.say(text, audio=...)`` needs no TTS plugin on the AgentSession.

    Streaming (when the TTS offers ``stream`` and ``streaming=True``): playback starts as soon
    as ``prebuffer_ms`` of audio has arrived. If the stream fails before any audio, the sink
    falls back once to unary ``synthesize``. If it fails after playback data began, the audio
    already delivered is played, the utterance is NOT restarted, and a TTS error is raised
    so the speech channel records ``speech_failed``.
    Cancellation (barge-in) interrupts playout and closes the stream.
    """

    def __init__(self, session: Any, tts: Any, frame_ms: int = 20, streaming: bool = True,
                 prebuffer_ms: int = 200, stream_sample_rate: int = 24000,
                 observer: Callable[..., None] | None = None) -> None:
        self._session = session
        self._tts = tts
        self._frame_ms = frame_ms
        self._streaming = streaming and hasattr(tts, "stream")
        self._prebuffer_ms = prebuffer_ms
        self._stream_rate = stream_sample_rate
        self.observer = observer  # optional latency marks, e.g. SessionRuntime.mark

    def _note(self, **fields: Any) -> None:
        if self.observer is not None:
            self.observer(**fields)

    async def say(self, text: str) -> bool:
        if self._streaming:
            return await self._say_streaming(text)
        return await self._say_unary(text)

    async def _play(self, text: str, frames: AsyncIterator[Any]) -> bool:
        handle = self._session.say(text, audio=frames, allow_interruptions=True, add_to_chat_ctx=False)
        try:
            await handle.wait_for_playout()
        except asyncio.CancelledError:
            handle.interrupt()
            raise
        return bool(handle.interrupted)

    async def _say_unary(self, text: str) -> bool:
        audio = await self._tts.synthesize(text)
        self._note(name="tts_first_audio", mode="unary", bytes=len(audio.data))
        return await self._play(text, self._frames(audio))

    async def _say_streaming(self, text: str) -> bool:
        from fdagent.providers.tts_base import TTSError, TTSStreamNotStarted

        chunks = self._tts.stream(text)
        try:
            first = await chunks.__anext__()
        except (TTSStreamNotStarted, StopAsyncIteration) as e:
            await chunks.aclose()
            self._note(name="tts_stream_fallback", reason=f"{type(e).__name__}: {e}"[:200])
            return await self._say_unary(text)
        self._note(name="tts_first_audio", mode="stream", bytes=len(first))
        failure: list[str] = []
        try:
            interrupted = await self._play(text, self._stream_frames(first, chunks, failure))
        finally:
            try:
                await chunks.aclose()  # stop the HTTP stream (barge-in / end of utterance)
            except RuntimeError:
                pass  # still suspended inside the frame generator; it is finalized with it
        if failure:
            raise TTSError(failure[0])
        return interrupted

    async def _stream_frames(self, first: bytes, chunks: AsyncIterator[bytes], failure: list[str]):
        """PCM chunks → fixed 20 ms frames, after a small prebuffer. A mid-stream error ends the
        audio gracefully (what was delivered still plays) and is reported via ``failure``."""
        from livekit import rtc

        # The backend declares its stream rate (Gemini 24 kHz, Piper 22.05 kHz); default otherwise.
        rate, ch = int(getattr(self._tts, "stream_sample_rate", None) or self._stream_rate), 1
        frame_bytes = max(1, rate * self._frame_ms // 1000) * 2 * ch
        prebuffer = rate * self._prebuffer_ms // 1000 * 2 * ch
        buf = bytearray(first)
        pending = True

        async def pull() -> bool:
            nonlocal pending
            try:
                buf.extend(await chunks.__anext__())
                return True
            except StopAsyncIteration:
                pending = False
            except Exception as e:  # TTSError / transport error after playback began: no restart
                failure.append(str(e) or type(e).__name__)
                pending = False
            return False

        while pending and len(buf) < prebuffer:
            await pull()
        while True:
            while len(buf) >= frame_bytes:
                yield rtc.AudioFrame(bytes(buf[:frame_bytes]), rate, ch, frame_bytes // (2 * ch))
                del buf[:frame_bytes]
            if not pending or not await pull():
                break
        tail = len(buf) - (len(buf) % (2 * ch))
        if tail:
            yield rtc.AudioFrame(bytes(buf[:tail]), rate, ch, tail // (2 * ch))

    async def _frames(self, audio: Any):
        from livekit import rtc

        samples = max(1, audio.sample_rate * self._frame_ms // 1000)
        step = samples * audio.channels * 2
        for i in range(0, len(audio.data), step):
            chunk = audio.data[i : i + step]
            yield rtc.AudioFrame(chunk, audio.sample_rate, audio.channels, len(chunk) // (2 * audio.channels))
