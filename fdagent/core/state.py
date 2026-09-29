"""Authoritative session state. Only the kernel (and functions it calls within a step) writes it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import DesiredCall, Operation, OpStatus, ToolSpec


@dataclass
class PendingReply:
    request_id: str
    generation: int
    text: str
    kind: str


@dataclass
class SessionState:
    session_id: str
    tools: dict[str, ToolSpec]

    # Intent tracking
    generation: int = 0  # bumps on every speech onset / meaningful final transcript
    speech_epoch: int = 0  # bumps only on user speech onset
    user_speaking: bool = False
    turn_open: bool = False  # user has spoken since the last stable point
    stable: bool = False  # current generation passed the stability window
    transcript: list[dict[str, Any]] = field(default_factory=list)  # {"generation", "text"}
    partial: str = ""

    # Plan (from the latest admitted reasoner proposal)
    desired: dict[str, DesiredCall] = field(default_factory=dict)
    desired_generation: int = -1
    invalid: dict[str, str] = field(default_factory=dict)  # key -> schema error

    # Operation ledger
    ops: dict[str, Operation] = field(default_factory=dict)  # call_id -> op
    op_by_key: dict[str, str] = field(default_factory=dict)  # key -> call_id
    orphan_effects: list[dict[str, Any]] = field(default_factory=list)

    # Reasoning bookkeeping
    latest_request_id: str | None = None
    latest_request_generation: int = -1

    # Speech bookkeeping
    agent_speaking: bool = False
    pending_reply: PendingReply | None = None
    spoken_requests: set[str] = field(default_factory=set)
    replied_generation: int = -1
    acked_generation: int = -1
    announced_failures: set[str] = field(default_factory=set)

    # Observability: last gate reason per key (journal only records changes)
    gate_blocks: dict[str, str] = field(default_factory=dict)

    def op_for_key(self, key: str) -> Operation | None:
        cid = self.op_by_key.get(key)
        return self.ops.get(cid) if cid else None

    def desired_in_order(self) -> list[DesiredCall]:
        return sorted(self.desired.values(), key=lambda d: d.index)

    def in_flight(self) -> list[Operation]:
        return [
            op
            for op in self.ops.values()
            if op.status == OpStatus.DISPATCHED and op.key in self.desired
        ]
