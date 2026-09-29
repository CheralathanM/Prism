"""Internal event types. Everything that can change session state arrives as one of these.

Events are immutable facts about something that happened (the user spoke, a tool returned,
a timer fired). They carry the identity metadata needed to judge whether they are still
relevant: generation, request_id, call_id, attempt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Event:
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = type(self).__name__
        return d


# ── User / perception ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UserSpeechStarted(Event):
    pass


@dataclass(frozen=True)
class UserTranscript(Event):
    text: str
    final: bool = True


@dataclass(frozen=True)
class UserTurnEnded(Event):
    pass


# ── Agent speech feedback (from the TTS / playout side) ──────────────────────
@dataclass(frozen=True)
class AgentSpeechStarted(Event):
    pass


@dataclass(frozen=True)
class AgentSpeechEnded(Event):
    interrupted: bool = False


# ── Reasoner (slow path) ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProposedCall:
    """A tool call the planner believes the current intent requires.

    ``args`` values may contain a dependency reference ``{"$ref": [dep_index, *path]}``
    pointing at the result of another call in the *same* proposal; it is resolved at
    dispatch time from the admitted result. ``occurrence`` distinguishes deliberate
    repeats of an identical call; identical calls with the same occurrence collapse.
    """

    tool: str
    args: dict[str, Any]
    depends_on: tuple[int, ...] = ()
    occurrence: int = 0


@dataclass(frozen=True)
class ReasonerProposal(Event):
    """Complete desired call set for ``generation``, plus optional speech.

    reply_kind: "final" (answers the user; only spoken once every desired call has an
    admitted success), "progress" (interim, never claims completion), or "clarify".
    """

    request_id: str
    generation: int
    calls: tuple[ProposedCall, ...] = ()
    reply: str | None = None
    reply_kind: str = "final"


@dataclass(frozen=True)
class ReasonerFailed(Event):
    request_id: str
    generation: int
    error: str


# ── Tools ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ToolResult(Event):
    """Outcome of one attempt of one operation.

    ``effect_applied`` lets a backend report that a state change happened even though
    the call is reported as failed (e.g. response lost after commit).
    """

    call_id: str
    attempt: int
    ok: bool
    payload: Any = None
    error: str | None = None
    retriable: bool = False
    effect_applied: bool | None = None


# ── Timers ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TimerFired(Event):
    timer_id: str
    kind: str  # "stability" | "tool_timeout"
    generation: int = 0
    call_id: str | None = None
    attempt: int = 0
    # Stability timers are keyed by speech epoch (bumped only on speech onset), not by
    # intent generation: a late STT final must not invalidate a silence window that has
    # already elapsed.
    epoch: int = 0


EVENT_TYPES: dict[str, type[Event]] = {
    cls.__name__: cls
    for cls in (
        UserSpeechStarted,
        UserTranscript,
        UserTurnEnded,
        AgentSpeechStarted,
        AgentSpeechEnded,
        ReasonerProposal,
        ReasonerFailed,
        ToolResult,
        TimerFired,
    )
}


def event_from_dict(d: dict[str, Any]) -> Event:
    """Inverse of ``Event.to_dict`` (used by journal replay)."""
    d = dict(d)
    cls = EVENT_TYPES[d.pop("type")]
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in names}
    if cls is ReasonerProposal:
        kwargs["calls"] = tuple(
            ProposedCall(
                tool=c["tool"],
                args=c["args"],
                depends_on=tuple(c.get("depends_on", ())),
                occurrence=c.get("occurrence", 0),
            )
            for c in kwargs.get("calls", ())
        )
    return cls(**kwargs)
