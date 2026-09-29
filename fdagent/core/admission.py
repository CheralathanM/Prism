"""Result admission: every tool outcome is recorded; only valid ones change current state."""

from __future__ import annotations

from dataclasses import dataclass

from .events import ToolResult
from .model import Operation, OpStatus
from .state import SessionState


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str
    op: Operation | None


def admit_result(state: SessionState, res: ToolResult) -> Admission:
    """Decide whether ``res`` may update the operation ledger.

    Rejections (result is logged by the journal, state untouched except bookkeeping):
      unknown_call      - no such call_id in this session
      stale_attempt     - result of an older attempt of a retried operation
      superseded        - the operation was logically cancelled; outcome stored as
                          ``late_outcome`` (and as an orphan effect if it changed state)
      duplicate_result  - the operation already has a terminal outcome
    """
    op = state.ops.get(res.call_id)
    if op is None:
        return Admission(False, "unknown_call", None)

    outcome = {
        "attempt": res.attempt,
        "ok": res.ok,
        "payload": res.payload,
        "error": res.error,
        "effect_applied": res.effect_applied,
    }
    op.history.append(outcome)

    if res.attempt != op.attempt:
        return Admission(False, "stale_attempt", op)

    if op.status == OpStatus.SUPERSEDED:
        op.late_outcome = outcome
        if op.state_changing and (res.ok or res.effect_applied):
            state.orphan_effects.append(
                {"call_id": op.call_id, "key": op.key, "tool": op.tool, "args": op.args, "payload": res.payload}
            )
        return Admission(False, "superseded", op)

    if op.status in (OpStatus.SUCCEEDED, OpStatus.FAILED):
        return Admission(False, "duplicate_result", op)

    # DISPATCHED, or UNKNOWN (a late answer resolves the uncertainty).
    if res.ok:
        op.status = OpStatus.SUCCEEDED
        op.result = res.payload
        op.error = None
    else:
        op.status = OpStatus.FAILED
        op.error = res.error or "unknown error"
        op.retriable = res.retriable
        op.effect_applied = res.effect_applied
    return Admission(True, "admitted", op)
