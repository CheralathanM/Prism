"""Actions emitted by the kernel. The runtime executes them; the kernel never does I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Action:
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = type(self).__name__
        return d


@dataclass(frozen=True)
class Speak(Action):
    text: str
    kind: str  # "backchannel" | "progress" | "final" | "failure" | "fallback"
    generation: int


@dataclass(frozen=True)
class StopSpeaking(Action):
    reason: str


@dataclass(frozen=True)
class DiscardStaleSpeech(Action):
    """Drop queued (not yet started) utterances planned for a generation older than
    ``below_generation``. Speech already playing is left to barge-in handling."""

    below_generation: int
    reason: str


@dataclass(frozen=True)
class RequestReasoning(Action):
    request_id: str
    generation: int
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class DispatchTool(Action):
    call_id: str
    attempt: int
    tool: str
    args: dict[str, Any]
    key: str
    idempotency_key: str
    generation: int
    state_changing: bool


@dataclass(frozen=True)
class AdvisoryCancel(Action):
    """Best-effort cancel. The call is already logically cancelled (SUPERSEDED); the
    backend may still finish, and its result will be logged but never used."""

    call_id: str
    reason: str


@dataclass(frozen=True)
class StartTimer(Action):
    timer_id: str
    kind: str
    delay_s: float
    generation: int = 0
    call_id: str | None = None
    attempt: int = 0
    epoch: int = 0
