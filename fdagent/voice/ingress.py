"""Protocol adapter · ingress: LiveKit session callbacks → internal events.

Pure translation, no LiveKit import, so it is unit-testable. VAD state gives speech
onset/offset; STT gives transcript segments. End-of-turn here is the raw VAD offset; the
kernel's stability window decides when intent is actually stable.
"""

from __future__ import annotations

from typing import Callable

from fdagent.core.events import Event, UserSpeechStarted, UserTranscript, UserTurnEnded


class IngressBridge:
    def __init__(self, post: Callable[[Event], object], on_final: Callable[[str], None] | None = None) -> None:
        self._post = post
        self._on_final = on_final
        self._speaking = False

    def on_user_state(self, new_state: str) -> None:
        if new_state == "speaking":
            if not self._speaking:
                self._speaking = True
                self._post(UserSpeechStarted())
        elif self._speaking:  # "listening" / "away"
            self._speaking = False
            self._post(UserTurnEnded())

    def on_transcript(self, text: str, is_final: bool) -> None:
        if is_final:
            if not text.strip():
                # Still forwarded: an empty final settles its speech segment in the kernel
                # (otherwise the segment would block dispatch until its deadline).
                self._post(UserTranscript("", final=True))
                return
            if self._on_final:
                self._on_final(text)
        self._post(UserTranscript(text, final=is_final))
