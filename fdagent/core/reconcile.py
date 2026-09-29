"""Reconcile desired calls against the operation ledger.

Called only from inside ``SessionKernel.step`` (so the kernel remains the single writer).
Rules, per desired-call key:
  key gone        → unresolved op becomes SUPERSEDED + AdvisoryCancel
  same key        → keep / reuse (never a second create)
  key back again  → revive the superseded op instead of creating a duplicate
  new key         → gate → schema check → DispatchTool
  failed op       → retry with the SAME call_id and idempotency key, only when safe
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from .actions import Action, AdvisoryCancel, DispatchTool, StartTimer
from .config import KernelConfig
from .gate import gate_check
from .ids import IdGen
from .model import UNRESOLVED, DesiredCall, Operation, OpStatus, is_ref, resolve_path
from .state import SessionState

Note = Callable[..., None]


def resolve_args(state: SessionState, dc: DesiredCall) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in dc.args.items():
        if is_ref(v):
            dep_key, *path = v["$ref"]
            op = state.op_for_key(dep_key)
            if op is None or op.status != OpStatus.SUCCEEDED:
                raise KeyError(f"dependency {dep_key} not satisfied")
            out[k] = copy.deepcopy(resolve_path(op.result, path))
        else:
            out[k] = copy.deepcopy(v)
    return out


def _retry_safe(op: Operation, config: KernelConfig, idempotent_backend: bool) -> bool:
    if op.attempt >= config.max_attempts:
        return False
    if op.status == OpStatus.UNKNOWN:
        # Timed out: we do not know whether the effect happened.
        return not op.state_changing or idempotent_backend
    if op.status == OpStatus.FAILED and op.retriable:
        if not op.state_changing:
            return True
        # State change: only retry if we know nothing was applied, or the backend
        # deduplicates on the idempotency key.
        return op.effect_applied is False or idempotent_backend
    return False


def terminal_failure(state: SessionState, key: str, config: KernelConfig) -> str | None:
    """Return a reason if the desired call ``key`` can no longer succeed, else None."""
    if key in state.invalid:
        return f"invalid arguments: {state.invalid[key]}"
    op = state.op_for_key(key)
    if op is None:
        return None
    spec = state.tools.get(op.tool)
    idem = bool(spec and spec.idempotent)
    if op.status == OpStatus.FAILED and not _retry_safe(op, config, idem):
        return op.error or "failed"
    if op.status == OpStatus.UNKNOWN and not _retry_safe(op, config, idem):
        return "no confirmation from the service (timed out)"
    return None


def reconcile(state: SessionState, config: KernelConfig, ids: IdGen, note: Note) -> list[Action]:
    actions: list[Action] = []

    # 1) Supersede unresolved operations whose key is no longer desired.
    for op in state.ops.values():
        if op.status in UNRESOLVED and op.key not in state.desired:
            op.status = OpStatus.SUPERSEDED
            actions.append(AdvisoryCancel(op.call_id, "key_no_longer_desired"))
            note("superseded", call_id=op.call_id, key=op.key, tool=op.tool, reason="key_no_longer_desired")

    # 2) Walk the plan in chain order.
    for dc in state.desired_in_order():
        spec = state.tools.get(dc.tool)
        op = state.op_for_key(dc.key)

        if op is not None:
            if op.status == OpStatus.SUPERSEDED:
                # The user came back to an intent we had cancelled. Revive, never duplicate.
                late = op.late_outcome
                if late is None:
                    op.status = OpStatus.DISPATCHED
                elif late["ok"]:
                    op.status, op.result = OpStatus.SUCCEEDED, late["payload"]
                else:
                    op.status, op.error = OpStatus.FAILED, late["error"]
                op.generation = dc.generation
                note("revived", call_id=op.call_id, key=op.key, status=op.status.value)
                continue
            if op.status in (OpStatus.DISPATCHED, OpStatus.SUCCEEDED):
                continue
            if not _retry_safe(op, config, bool(spec and spec.idempotent)):
                continue
            reason = gate_check(state, dc, spec, config)
            if reason:
                _block(state, dc, reason, note)
                continue
            op.attempt += 1
            op.status = OpStatus.DISPATCHED
            op.generation = dc.generation
            actions += _dispatch(op, spec)  # type: ignore[arg-type]
            note("retry", call_id=op.call_id, key=op.key, attempt=op.attempt, idempotency_key=op.call_id)
            continue

        if dc.key in state.invalid:
            continue
        reason = gate_check(state, dc, spec, config)
        if reason:
            _block(state, dc, reason, note)
            continue
        assert spec is not None
        try:
            args = resolve_args(state, dc)
        except (KeyError, IndexError, TypeError) as e:
            state.invalid[dc.key] = f"unresolvable reference: {e}"
            note("invalid_call", key=dc.key, tool=dc.tool, error=state.invalid[dc.key])
            continue
        err = spec.validate(args)
        if err:
            state.invalid[dc.key] = err
            note("invalid_call", key=dc.key, tool=dc.tool, error=err)
            continue

        op = Operation(
            call_id=ids.next("call"),
            key=dc.key,
            tool=dc.tool,
            args=args,
            state_changing=spec.state_changing,
            generation=dc.generation,
            parent_keys=dc.depends_on,
        )
        state.ops[op.call_id] = op
        state.op_by_key[dc.key] = op.call_id
        state.gate_blocks.pop(dc.key, None)
        actions += _dispatch(op, spec)
        note(
            "dispatch",
            call_id=op.call_id,
            key=op.key,
            tool=op.tool,
            args=op.args,
            generation=op.generation,
            parents=list(op.parent_keys),
            state_changing=op.state_changing,
        )
    return actions


def _dispatch(op: Operation, spec) -> list[Action]:
    return [
        DispatchTool(
            call_id=op.call_id,
            attempt=op.attempt,
            tool=op.tool,
            args=copy.deepcopy(op.args),
            key=op.key,
            idempotency_key=op.call_id,  # stable across retries of the same operation
            generation=op.generation,
            state_changing=op.state_changing,
        ),
        StartTimer(
            timer_id=f"timeout:{op.call_id}:{op.attempt}",
            kind="tool_timeout",
            delay_s=spec.timeout_s,
            generation=op.generation,
            call_id=op.call_id,
            attempt=op.attempt,
        ),
    ]


def _block(state: SessionState, dc: DesiredCall, reason: str, note: Note) -> None:
    if state.gate_blocks.get(dc.key) != reason:
        state.gate_blocks[dc.key] = reason
        note("gate_blocked", key=dc.key, tool=dc.tool, reason=reason)
