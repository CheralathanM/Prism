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
