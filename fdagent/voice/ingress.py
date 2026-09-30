"""Protocol adapter · ingress: LiveKit session callbacks → internal events.

Pure translation, no LiveKit import, so it is unit-testable. VAD state gives speech
onset/offset; STT gives transcript segments. End-of-turn here is the raw VAD offset; the
kernel's stability window decides when intent is actually stable.

Transcripts come from one of two sources, never both:
  * ``on_transcript`` — LiveKit's ``user_input_transcribed`` events (hosted STT plugins);
  * ``on_stt_event`` — our local STT adapter, which reports every segment's lifecycle
    (started / final / failed), so the kernel knows a segment is still being transcribed.
With ``livekit_transcripts=False`` the LiveKit events are ignored to avoid double counting.
"""

from __future__ import annotations

from typing import Any, Callable

from fdagent.core.events import (
    Event,
    TranscriptionFailed,
    TranscriptionStarted,
    UserSpeechStarted,
    UserTranscript,
    UserTurnEnded,
)


class IngressBridge:
    def __init__(self, post: Callable[[Event], object], on_final: Callable[[str], None] | None = None,
                 livekit_transcripts: bool = True) -> None:
        self._post = post
        self._on_final = on_final
        self._speaking = False
        self.livekit_transcripts = livekit_transcripts

    def on_user_state(self, new_state: str) -> None:
        if new_state == "speaking":
            if not self._speaking:
                self._speaking = True
                self._post(UserSpeechStarted())
        elif self._speaking:  # "listening" / "away"
            self._speaking = False
            self._post(UserTurnEnded())

    def on_transcript(self, text: str, is_final: bool) -> None:
        if not self.livekit_transcripts:
            return
        self._final_or_partial(text, is_final)

    def on_stt_event(self, kind: str, **fields: Any) -> None:
        """Lifecycle callback from the local STT adapter."""
        if kind == "started":
            self._post(TranscriptionStarted())
        elif kind == "final":
            self._final_or_partial(fields.get("text", ""), True)
        elif kind == "failed":
            self._post(TranscriptionFailed(error=str(fields.get("error", ""))))

    def _final_or_partial(self, text: str, is_final: bool) -> None:
        if is_final:
            if not text.strip():
                # Still forwarded: an empty final settles its speech segment in the kernel
                # (otherwise the segment would block dispatch until its deadline).
                self._post(UserTranscript("", final=True))
                return
            if self._on_final:
                self._on_final(text)
        self._post(UserTranscript(text, final=is_final))
