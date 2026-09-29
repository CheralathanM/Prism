"""Tool specs, desired calls, operations, and desired-call-key computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ToolParam:
    name: str
    type: str  # "string" | "number" | "integer" | "boolean"
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: tuple[ToolParam, ...]
    state_changing: bool = False
    # True if the backend deduplicates on the idempotency key, which makes retrying a
    # state change with an unknown outcome safe.
    idempotent: bool = False
    timeout_s: float = 15.0

    def validate(self, args: dict[str, Any]) -> str | None:
        """Return an error string, or None if args are acceptable."""
        known = {p.name for p in self.params}
        for extra in args:
            if extra not in known:
                return f"unknown argument '{extra}'"
        for p in self.params:
            if p.name not in args or args[p.name] is None:
                if p.required:
                    return f"missing required argument '{p.name}'"
                continue
            v = args[p.name]
            if p.type == "string" and not isinstance(v, str):
                return f"'{p.name}' must be a string"
            if p.type in ("number", "integer") and (isinstance(v, bool) or not isinstance(v, (int, float))):
                return f"'{p.name}' must be a number"
            if p.type == "boolean" and not isinstance(v, bool):
                return f"'{p.name}' must be a boolean"
        return None


class OpStatus(str, Enum):
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"  # timed out; effect may or may not have happened
    SUPERSEDED = "superseded"  # logically cancelled; any result is never used


UNRESOLVED = (OpStatus.DISPATCHED, OpStatus.UNKNOWN)


@dataclass
class DesiredCall:
    key: str
    tool: str
    args: dict[str, Any]  # may contain {"$ref": [...]} rewritten to dependency keys
    depends_on: tuple[str, ...]  # dependency keys
    generation: int
    index: int  # position in the proposal (chain order)
    occurrence: int = 0


@dataclass
class Operation:
    call_id: str
    key: str
    tool: str
    args: dict[str, Any]  # resolved args actually sent
    state_changing: bool
    generation: int
    parent_keys: tuple[str, ...]
    status: OpStatus = OpStatus.DISPATCHED
    attempt: int = 1
    result: Any = None
    error: str | None = None
    retriable: bool = False
    effect_applied: bool | None = None
    late_outcome: dict[str, Any] | None = None  # result that arrived after SUPERSEDED
    history: list[dict[str, Any]] = field(default_factory=list)


# ── Desired call key ─────────────────────────────────────────────────────────
def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    if isinstance(v, dict):
        return {k: _norm(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return str(v)


def desired_call_key(tool: str, args: dict[str, Any], occurrence: int = 0) -> str:
    """Stable identity of *what* is being asked for, independent of when or by whom.

    Same tool + semantically-equal args + same occurrence → same key → the kernel reuses
    the existing operation instead of creating a second one. Dependency references must
    already be rewritten to ``{"$ref": [dep_key, *path]}`` so the key captures lineage.
    """
    canon = json.dumps(_norm({k: v for k, v in args.items() if v is not None}), sort_keys=True)
    digest = hashlib.sha1(canon.encode()).hexdigest()[:10]
    return f"{tool}:{digest}#{occurrence}"


def is_ref(v: Any) -> bool:
    return isinstance(v, dict) and set(v) == {"$ref"}


def resolve_path(payload: Any, path: list[Any]) -> Any:
    cur = payload
    for p in path:
        if isinstance(cur, dict):
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(p)
    return cur
