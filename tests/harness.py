"""Deterministic test harness: drive the kernel event-by-event with a fake clock."""

from __future__ import annotations

from typing import Any

from fdagent.core.actions import Action, DispatchTool, RequestReasoning, Speak, StartTimer
from fdagent.core.config import KernelConfig
from fdagent.core.events import (
    Event,
    ProposedCall,
    ReasonerProposal,
    TimerFired,
    ToolResult,
    UserSpeechStarted,
    UserTranscript,
    UserTurnEnded,
)
from fdagent.core.inbox import Envelope
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.core.model import OpStatus, ToolParam, ToolSpec

S = lambda n: ToolParam(n, "string")  # noqa: E731

TOOLS = [
    ToolSpec("search_flights", "Search flights", (S("destination"), S("date"))),
    ToolSpec("book_flight", "Book a flight", (S("flight_id"), S("passenger_name")), state_changing=True),
    ToolSpec("send_confirmation", "Send booking confirmation", (S("booking_ref"),), state_changing=True),
    ToolSpec("book_flight_idem", "Book (idempotent backend)", (S("flight_id"), S("passenger_name")),
             state_changing=True, idempotent=True),
]


def ref(index: int, *path: Any) -> dict:
    return {"$ref": [index, *path]}


class Harness:
    def __init__(self, tools=TOOLS, **cfg: Any) -> None:
        self.journal = Journal()
        self.k = SessionKernel("test-session", list(tools), KernelConfig(**cfg), journal=self.journal)
        self.seq = 0
        self.t = 0.0
        self.log: list[Action] = []

    # ── low level ──
    def send(self, ev: Event, dt: float = 0.01) -> list[Action]:
        self.seq += 1
        self.t += dt
        acts = self.k.step(Envelope(self.seq, self.t, ev))
        self.log += acts
        return acts

    @property
    def s(self):
        return self.k.state

    # ── user side ──
    def user_says(self, text: str, end_turn: bool = True) -> list[Action]:
        acts = self.send(UserSpeechStarted())
        acts += self.send(UserTranscript(text, final=True))
        if end_turn:
            acts += self.send(UserTurnEnded())
        return acts

    def stabilize(self) -> list[Action]:
        return self.send(TimerFired(f"stability:{self.s.speech_epoch}", "stability", epoch=self.s.speech_epoch), dt=0.6)

    # ── reasoner side ──
    def latest_request(self) -> RequestReasoning:
        return [a for a in self.log if isinstance(a, RequestReasoning)][-1]

    def propose(self, calls: list[ProposedCall], reply: str | None = None, kind: str = "final",
                request: RequestReasoning | None = None) -> list[Action]:
        r = request or self.latest_request()
        return self.send(ReasonerProposal(r.request_id, r.generation, tuple(calls), reply, kind))

    # ── tool side ──
    def result(self, call_id: str, ok: bool = True, payload: Any = None, attempt: int = 1, **kw: Any) -> list[Action]:
        return self.send(ToolResult(call_id, attempt, ok, payload, **kw))

    # ── queries ──
    def dispatches(self, acts: list[Action] | None = None) -> list[DispatchTool]:
        return [a for a in (self.log if acts is None else acts) if isinstance(a, DispatchTool)]

    def speech(self, acts: list[Action] | None = None) -> list[Speak]:
        return [a for a in (self.log if acts is None else acts) if isinstance(a, Speak)]

    def decisions(self, kind: str) -> list[dict]:
        return [d for r in self.journal.records if r["kind"] == "step" for d in r["decisions"] if d["kind"] == kind]

    def op(self, call_id: str):
        return self.s.ops[call_id]


__all__ = ["Harness", "ProposedCall", "ref", "OpStatus", "StartTimer", "TOOLS"]
