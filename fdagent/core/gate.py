"""Commit barrier: decides whether a desired call may leave the kernel *now*.

The gate never says "no forever"; it says "not yet, because <reason>". Reconcile runs
after every event, so a blocked call is re-checked as soon as anything changes.
Duplicate protection (existing op for the key) lives in reconcile, which owns the ledger.
"""

from __future__ import annotations

from .config import KernelConfig
from .model import DesiredCall, OpStatus, ToolSpec
from .state import SessionState


def gate_check(
    state: SessionState, dc: DesiredCall, spec: ToolSpec | None, config: KernelConfig
) -> str | None:
    """Return None if dispatch is allowed, else a machine-readable reason."""
    if spec is None:
        return "unknown_tool"
    # A plan computed for an older generation must not execute while a fresher plan is
    # being computed, even if the old plan's calls look fine.
    if dc.generation != state.generation:
        return "stale_plan"
    if spec.state_changing or config.gate_read_only:
        if state.user_speaking:
            return "user_speaking"
        if not state.stable:
            return "intent_not_stable"
        # Silence alone is not enough: words already spoken may still be in the STT pipeline
        # (e.g. an argument said in a later segment) and could change the call.
        if state.pending_transcripts:
            return "transcript_pending"
    for dep in dc.depends_on:
        op = state.op_for_key(dep)
        if op is None or op.status != OpStatus.SUCCEEDED:
            return "waiting_dependency"
    return None
