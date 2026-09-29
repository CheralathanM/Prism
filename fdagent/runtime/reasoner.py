"""Reasoner (slow path) interface.

A reasoner receives a read-only snapshot and returns a Draft: the complete set of tool
calls the current intent requires (including already-executed ones that are still
right), plus optional speech. It never sees ids it could act on and never touches state.
The OpenAI implementation lands in Phase 3 (``openai_reasoner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fdagent.core.actions import RequestReasoning
from fdagent.core.events import ProposedCall


@dataclass(frozen=True)
class Draft:
    calls: tuple[ProposedCall, ...] = ()
    reply: str | None = None
    reply_kind: str = "final"


class Reasoner(Protocol):
    async def propose(self, request: RequestReasoning) -> Draft: ...
