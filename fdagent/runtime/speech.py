"""Speech egress: one ordered playout channel per session.

Utterances play strictly in order. ``stop()`` (barge-in) interrupts the current utterance
and drops anything queued. Playout start/end are reported back as events so the kernel
knows whether the agent is speaking.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Protocol

from fdagent.core.actions import Speak
from fdagent.core.events import AgentSpeechEnded, AgentSpeechStarted, Event


class SpeechSink(Protocol):
    async def say(self, text: str) -> bool | None:
        """Play ``text``; return when playout finished. Return True if the transport
        interrupted playout on its own (e.g. LiveKit barge-in). Cancellation = interrupted."""
        ...


class SpeechChannel:
    def __init__(self, sink: SpeechSink, post: Callable[[Event], object], mark: Callable[..., None]) -> None:
        self._sink = sink
        self._post = post
        self._mark = mark
        self._queue: asyncio.Queue[Speak] = asyncio.Queue()
        self._current: asyncio.Task | None = None
        self._interrupt_requested = False
        self._worker = asyncio.create_task(self._run(), name="speech")

    def enqueue(self, a: Speak) -> None:
        self._queue.put_nowait(a)

    def stop(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
        if self._current and not self._current.done():
            self._interrupt_requested = True
            self._current.cancel()

    async def _run(self) -> None:
        while True:
            a = await self._queue.get()
            self._post(AgentSpeechStarted())
            self._mark(name="speech_start", speech_kind=a.kind, generation=a.generation, text=a.text)
            self._current = asyncio.create_task(self._sink.say(a.text))
            interrupted = False
            try:
                interrupted = bool(await self._current)
            except asyncio.CancelledError:
                # Cancelling the worker (shutdown, or asyncio.run cleanup) also cancels the
                # awaited utterance. Only an explicit stop() counts as barge-in; anything
                # else must propagate or the worker becomes uncancellable.
                if not self._interrupt_requested:
                    raise
                interrupted = True
            except Exception as e:
                # A TTS/transport failure loses this utterance but must not kill the channel.
                self._mark(name="speech_failed", speech_kind=a.kind, error=f"{type(e).__name__}: {e}")
            finally:
                self._interrupt_requested = False
            self._post(AgentSpeechEnded(interrupted=interrupted))
            self._mark(name="speech_end", speech_kind=a.kind, interrupted=interrupted)

    async def aclose(self) -> None:
        # Not stop(): shutdown must not be mistaken for barge-in (see _run).
        while not self._queue.empty():
            self._queue.get_nowait()
        self._interrupt_requested = False
        self._worker.cancel()
        await asyncio.gather(self._worker, return_exceptions=True)
