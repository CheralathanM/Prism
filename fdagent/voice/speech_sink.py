"""Protocol adapter · egress (speech): kernel Speak actions → LiveKit TTS playout.

Duck-typed against ``AgentSession.say`` / ``SpeechHandle`` so it can be tested with fakes.
Speech is not added to LiveKit's chat context: the kernel owns conversation state.
"""

from __future__ import annotations

import asyncio
from typing import Any


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
    """Synthesize with an external TTS (e.g. Gemini), then play the PCM through LiveKit.

    ``session.say(text, audio=...)`` needs no TTS plugin on the AgentSession. Cancellation
    during synthesis (barge-in before audio exists) simply abandons the request.
    """

    def __init__(self, session: Any, tts: Any, frame_ms: int = 20) -> None:
        self._session = session
        self._tts = tts
        self._frame_ms = frame_ms

    async def say(self, text: str) -> bool:
        audio = await self._tts.synthesize(text)
        handle = self._session.say(text, audio=self._frames(audio), allow_interruptions=True, add_to_chat_ctx=False)
        try:
            await handle.wait_for_playout()
        except asyncio.CancelledError:
            handle.interrupt()
            raise
        return bool(handle.interrupted)

    async def _frames(self, audio: Any):
        from livekit import rtc

        samples = max(1, audio.sample_rate * self._frame_ms // 1000)
        step = samples * audio.channels * 2
        for i in range(0, len(audio.data), step):
            chunk = audio.data[i : i + step]
            yield rtc.AudioFrame(chunk, audio.sample_rate, audio.channels, len(chunk) // (2 * audio.channels))
