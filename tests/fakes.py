"""Controllable fakes for asyncio race tests (no network, no real models)."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable

from fdagent.adapters.tool_protocol import ToolTransientError
from fdagent.core.actions import RequestReasoning
from fdagent.core.config import KernelConfig
from fdagent.core.events import ProposedCall
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.runtime.loop import SessionRuntime
from fdagent.runtime.reasoner import Draft

from .harness import TOOLS

CITIES = ("Delhi", "Mumbai", "Rome", "Milan", "Oslo")


class CityReasoner:
    """Toy planner for tests: last city mentioned wins → search_flights(city, 'May 3').
    Replies once every planned call has succeeded."""

    def __init__(self, delay: Callable[[RequestReasoning], float] = lambda r: 0.0) -> None:
        self.delay = delay
        self.requests: list[RequestReasoning] = []

    async def propose(self, req: RequestReasoning) -> Draft:
        self.requests.append(req)
        await asyncio.sleep(self.delay(req))
        text = " ".join(req.snapshot["transcript"])
        found = [(m.start(), c) for c in CITIES for m in re.finditer(c, text)]
        if not found:
            return Draft(reply="Where would you like to go?", reply_kind="clarify")
        city = max(found)[1]
        calls = (ProposedCall("search_flights", {"destination": city, "date": "May 3"}),)
        done = req.snapshot["calls"] and all(c["status"] == "succeeded" for c in req.snapshot["calls"])
        return Draft(calls, reply=f"Here are flights to {city}." if done else None)


class GatedBackend:
    """Async backend; calls whose destination is in ``gates`` wait until released."""

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.started: list[tuple[str, dict]] = []
        self.effects = 0
        self.fail_first: set[str] = set()

    def gate(self, name: str) -> asyncio.Event:
        return self.gates.setdefault(name, asyncio.Event())

    async def acall(self, tool: str, args: dict[str, Any], idem: str) -> Any:
        self.started.append((tool, args))
        name = args.get("destination") or args.get("flight_id")
        if name in self.gates:
            await self.gates[name].wait()
        if tool in self.fail_first:
            self.fail_first.discard(tool)
            raise ToolTransientError("temporarily unavailable", effect_applied=False)
        if tool.startswith("book"):
            self.effects += 1
            return {"booking_ref": f"BR-{self.effects}"}
        return {"flights": [{"flight_id": f"FL-{name}"}]}


class BlockingBackend:
    """Synchronous backend that blocks its thread, like the harness's time.sleep latency."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def call(self, tool: str, args: dict[str, Any], idem: str) -> Any:
        time.sleep(self.seconds)
        return {"flights": [{"flight_id": "FL-X"}]}


class RecordingSink:
    def __init__(self, seconds: float = 0.01) -> None:
        self.said: list[str] = []
        self.seconds = seconds

    async def say(self, text: str) -> None:
        self.said.append(text)
        await asyncio.sleep(self.seconds)


def make_runtime(reasoner=None, backend=None, sink=None, tools=TOOLS, session_id="rt", **cfg) -> SessionRuntime:
    cfg.setdefault("stability_s", 0.05)
    kernel = SessionKernel(session_id, list(tools), KernelConfig(**cfg), journal=Journal())
    return SessionRuntime(kernel, reasoner or CityReasoner(), backend or GatedBackend(), sink or RecordingSink())


async def until(pred: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


def decisions(rt: SessionRuntime, kind: str) -> list[dict]:
    return [d for r in rt.kernel.journal.records if r["kind"] == "step" for d in r["decisions"] if d["kind"] == kind]


def marks(rt: SessionRuntime, name: str) -> list[dict]:
    return [r for r in rt.kernel.journal.records if r["kind"] == "note" and r.get("name") == name]
