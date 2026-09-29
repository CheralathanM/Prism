"""Session runtime: drain the inbox, step the kernel, execute actions asynchronously.

    ingress ──post()──► Inbox ──► kernel.step() ──► actions
                          ▲                           │
                          └── ToolResult / ReasonerProposal / TimerFired / AgentSpeech*

The loop body is synchronous per event (``kernel.step`` never awaits), and every action is
started as its own task, so a slow model, tool, or TTS can never delay the next event
(INVARIANT 1). One runtime = one conversation; nothing is shared between runtimes
(INVARIANT 12).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from fdagent.core.actions import (
    Action,
    AdvisoryCancel,
    DispatchTool,
    RequestReasoning,
    Speak,
    StartTimer,
    StopSpeaking,
)
from fdagent.core.events import Event, ReasonerFailed, ReasonerProposal
from fdagent.core.inbox import Envelope, Inbox
from fdagent.core.kernel import SessionKernel

from .reasoner import Reasoner
from .speech import SpeechChannel, SpeechSink
from .timers import Timers
from .tool_executor import ToolExecutor

log = logging.getLogger(__name__)


class SessionRuntime:
    def __init__(
        self,
        kernel: SessionKernel,
        reasoner: Reasoner,
        tool_backend: Any,
        speech_sink: SpeechSink,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.kernel = kernel
        self._clock = clock
        self.inbox = Inbox(clock)
        self._reasoner = reasoner
        self._tool_backend = tool_backend
        self._sink = speech_sink
        self._reasoning: asyncio.Task | None = None
        self._runner: asyncio.Task | None = None
        self._mark_listeners: list[Callable[[dict[str, Any]], None]] = []
        self.steps = 0
        self.failure: BaseException | None = None
        # Components needing a running loop are created in start().
        self.timers: Timers | None = None
        self.executor: ToolExecutor | None = None
        self.speech: SpeechChannel | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    async def start(self) -> None:
        self.timers = Timers(self.post)
        self.executor = ToolExecutor(self._tool_backend, self.post, self.mark)
        self.speech = SpeechChannel(self._sink, self.post, self.mark)
        self._runner = asyncio.create_task(self._run(), name="kernel-loop")

    async def aclose(self) -> None:
        if self._runner:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        if self._reasoning:
            self._reasoning.cancel()
            await asyncio.gather(self._reasoning, return_exceptions=True)
        for c in (self.timers, self.executor, self.speech):
            if c is not None:
                await c.aclose()
        if self.kernel.journal:
            self.kernel.journal.close()

    # ── ingress ─────────────────────────────────────────────────────────────
    def post(self, event: Event) -> Envelope:
        return self.inbox.put(event)

    def add_mark_listener(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._mark_listeners.append(fn)

    def mark(self, **fields: Any) -> None:
        """Latency / observability mark (journal note, not replayed)."""
        fields = {"t": self._clock(), "wall": time.time(), **fields}
        if self.kernel.journal:
            self.kernel.journal.annotate(**fields)
        for fn in self._mark_listeners:
            try:
                fn(fields)
            except Exception:  # telemetry must never break the conversation
                log.exception("mark listener failed")

    async def idle(self, settle_s: float = 0.02, timeout_s: float = 5.0) -> None:
        """Test/debug helper: wait until the inbox stays empty for ``settle_s``."""
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            if len(self.inbox) == 0:
                await asyncio.sleep(settle_s)
                if len(self.inbox) == 0:
                    return
            await asyncio.sleep(0.005)
        raise TimeoutError("runtime did not become idle")

    # ── loop ────────────────────────────────────────────────────────────────
    async def _run(self) -> None:
        while True:
            env = await self.inbox.get()
            try:
                actions = self.kernel.step(env)
            except Exception as e:
                # A kernel bug is fatal for the session: fail loudly rather than guess.
                self.failure = e
                log.exception("kernel step failed at seq %s", env.seq)
                raise
            self.steps += 1
            for a in actions:
                self._execute(a)

    def _execute(self, a: Action) -> None:
        if isinstance(a, DispatchTool):
            self.executor.submit(a)  # type: ignore[union-attr]
        elif isinstance(a, AdvisoryCancel):
            self.executor.cancel(a.call_id, a.reason)  # type: ignore[union-attr]
        elif isinstance(a, RequestReasoning):
            if self._reasoning and not self._reasoning.done():
                # Its proposal would be rejected anyway; stop spending on it.
                self._reasoning.cancel()
                self.mark(name="reasoning_cancelled", superseded_by=a.request_id)
            self._reasoning = asyncio.create_task(self._reason(a), name=f"reason:{a.request_id}")
        elif isinstance(a, StartTimer):
            self.timers.start(a)  # type: ignore[union-attr]
        elif isinstance(a, Speak):
            self.speech.enqueue(a)  # type: ignore[union-attr]
        elif isinstance(a, StopSpeaking):
            self.speech.stop()  # type: ignore[union-attr]
        else:
            raise TypeError(f"unhandled action {a!r}")

    async def _reason(self, a: RequestReasoning) -> None:
        self.mark(name="reasoning_start", request_id=a.request_id, generation=a.generation)
        try:
            draft = await self._reasoner.propose(a)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("reasoner failed")
            self.post(ReasonerFailed(a.request_id, a.generation, f"{type(e).__name__}: {e}"))
            return
        self.mark(name="reasoning_end", request_id=a.request_id)
        self.post(ReasonerProposal(a.request_id, a.generation, tuple(draft.calls), draft.reply, draft.reply_kind))
