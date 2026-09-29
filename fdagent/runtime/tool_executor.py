"""Tool executor: runs DispatchTool actions without ever blocking the event loop.

- Sync backends run in a worker thread (``asyncio.to_thread``); async backends are awaited.
- Outcomes are posted back to the inbox as ``ToolResult`` events. The executor never
  touches session state (INVARIANT 2).
- Idempotency guard for state-changing calls, keyed by the operation's idempotency key:
    * a second dispatch while the first attempt is still running attaches to it instead
      of calling the backend again;
    * a dispatch whose key already completed successfully replays the recorded outcome.
  Together with the kernel's retry rules this means an operation's effect is applied at
  most once per session even if a DispatchTool is delivered twice (INVARIANTS 7, 8).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from fdagent.adapters.tool_protocol import ToolTransientError
from fdagent.core.actions import DispatchTool
from fdagent.core.events import Event, ToolResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Outcome:
    ok: bool
    payload: Any
    error: str | None
    retriable: bool
    effect_applied: bool | None


class ToolExecutor:
    def __init__(
        self,
        backend: Any,
        post: Callable[[Event], Any],
        mark: Callable[..., None] | None = None,
    ) -> None:
        self._backend = backend
        self._post = post
        self._mark = mark or (lambda **_: None)
        self._tasks: dict[str, asyncio.Task] = {}  # call_id -> latest attempt task
        self._running: dict[str, asyncio.Task] = {}  # idempotency_key -> running state change
        self._completed: dict[str, _Outcome] = {}  # idempotency_key -> successful state change
        self.backend_invocations = 0

    def submit(self, d: DispatchTool) -> None:
        self._tasks[d.call_id] = asyncio.create_task(self._run(d), name=f"tool:{d.call_id}:{d.attempt}")

    def cancel(self, call_id: str, reason: str) -> None:
        """Advisory cancel. Thread-backed calls cannot be interrupted and state changes are
        never torn mid-flight; either way the kernel already ignores the result."""
        self._mark(name="tool_cancel_advisory", call_id=call_id, reason=reason)

    async def aclose(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _run(self, d: DispatchTool) -> None:
        key = d.idempotency_key
        if d.state_changing and key in self._completed:
            self._mark(name="tool_idempotent_replay", call_id=d.call_id, attempt=d.attempt)
            self._emit(d, self._completed[key])
            return
        if d.state_changing and key in self._running:
            self._mark(name="tool_attach_inflight", call_id=d.call_id, attempt=d.attempt)
            outcome = await asyncio.shield(self._running[key])
            self._emit(d, outcome)
            return

        task = asyncio.ensure_future(self._invoke(d))
        if d.state_changing:
            self._running[key] = task
        try:
            outcome = await task
        finally:
            self._running.pop(key, None)
        if d.state_changing and outcome.ok:
            self._completed[key] = outcome
        self._emit(d, outcome)

    async def _invoke(self, d: DispatchTool) -> _Outcome:
        self._mark(name="tool_start", call_id=d.call_id, attempt=d.attempt, tool=d.tool)
        self.backend_invocations += 1
        try:
            if hasattr(self._backend, "acall"):
                payload = await self._backend.acall(d.tool, d.args, d.idempotency_key)
            else:
                payload = await asyncio.to_thread(self._backend.call, d.tool, d.args, d.idempotency_key)
            if isinstance(payload, dict) and payload.get("status") == "error":
                out = _Outcome(False, payload, str(payload.get("message", "tool error")), False, None)
            else:
                out = _Outcome(True, payload, None, False, None)
        except ToolTransientError as e:
            out = _Outcome(False, None, str(e), True, e.effect_applied)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # explicit failure, never a silent guess
            log.exception("tool %s failed", d.tool)
            out = _Outcome(False, None, f"{type(e).__name__}: {e}", False, None)
        self._mark(name="tool_end", call_id=d.call_id, attempt=d.attempt, tool=d.tool, ok=out.ok)
        return out

    def _emit(self, d: DispatchTool, o: _Outcome) -> None:
        self._post(ToolResult(d.call_id, d.attempt, o.ok, o.payload, o.error, o.retriable, o.effect_applied))
