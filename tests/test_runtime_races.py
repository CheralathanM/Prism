"""Asyncio race tests: real runtime loop, real tasks and timers, fake models/tools."""

from __future__ import annotations

import asyncio
import time

from fdagent.core.actions import DispatchTool
from fdagent.core.events import ReasonerProposal, ProposedCall, ToolResult, UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.runtime.tool_executor import ToolExecutor

from .fakes import BlockingBackend, CityReasoner, GatedBackend, RecordingSink, decisions, make_runtime, marks, until
from .harness import ProposedCall as PC


def run(coro):
    return asyncio.run(coro)


def say(rt, text, end=True):
    rt.post(UserSpeechStarted())
    rt.post(UserTranscript(text))
    if end:
        rt.post(UserTurnEnded())


def test_R1_correction_while_tool_in_flight():
    """INVARIANTS 5, 6, 10: late Delhi result arrives after Mumbai completed; it is rejected
    and the spoken answer is about Mumbai only."""

    async def main():
        backend, sink = GatedBackend(), RecordingSink()
        delhi_gate = backend.gate("Delhi")
        rt = make_runtime(backend=backend, sink=sink)
        await rt.start()
        say(rt, "Find a flight to Delhi")
        await until(lambda: any(a.get("destination") == "Delhi" for _, a in backend.started))
        say(rt, "Actually, Mumbai")
        await until(lambda: "Here are flights to Mumbai." in sink.said)
        delhi_gate.set()
        await rt.idle()
        rejected = decisions(rt, "result_rejected")
        assert [r["reason"] for r in rejected] == ["superseded"]
        assert [c["args"]["destination"] for c in rt.kernel.snapshot()["calls"]] == ["Mumbai"]
        assert not any("Delhi" in s for s in sink.said)
        await rt.aclose()

    run(main())


def test_R2_events_keep_flowing_while_reasoner_is_slow():
    """INVARIANT 1: a 300 ms reasoner never delays event processing; outdated reasoning is
    cancelled and, if its proposal still arrives, rejected."""

    async def main():
        reasoner = CityReasoner(delay=lambda r: 0.3 if r.request_id == "req-1" else 0.0)
        backend = GatedBackend()
        rt = make_runtime(reasoner=reasoner, backend=backend)
        await rt.start()
        t0 = time.monotonic()
        say(rt, "I need a flight to Lisbon", end=False)
        rt.post(UserTranscript("no wait, Porto"))
        rt.post(UserTurnEnded())
        await until(lambda: rt.steps >= 4)
        assert time.monotonic() - t0 < 0.1  # processed while req-1 still "thinking"
        await until(lambda: len(backend.started) == 1)
        rt.post(ReasonerProposal("req-1", 1, (PC("search_flights", {"destination": "Lisbon", "date": "May 3"}),)))
        await rt.idle()
        assert [a["destination"] for _, a in backend.started] == ["Porto"]
        assert marks(rt, "reasoning_cancelled")
        assert decisions(rt, "proposal_rejected")[-1]["reason"] == "superseded_request"
        await rt.aclose()

    run(main())


def test_R3_blocking_tool_backend_does_not_block_the_loop():
    """INVARIANT 1: the harness mock sleeps synchronously; we run it in a thread, so user
    events are handled immediately while the tool is busy."""

    async def main():
        rt = make_runtime(backend=BlockingBackend(0.5))
        await rt.start()
        say(rt, "Find a flight to Oslo")
        await until(lambda: marks(rt, "tool_start"))
        before = rt.steps
        t0 = time.monotonic()
        rt.post(UserSpeechStarted())
        await until(lambda: rt.steps > before, timeout=0.2)
        assert time.monotonic() - t0 < 0.1
        await rt.aclose()

    run(main())


def test_R4_executor_never_runs_the_same_state_change_twice():
    """INVARIANT 7: a DispatchTool delivered twice (in flight, then after completion) reaches
    the backend once; every delivery still gets a result event."""

    async def main():
        backend = GatedBackend()
        gate = backend.gate("FL1")
        posted: list[ToolResult] = []
        ex = ToolExecutor(backend, posted.append)
        d = DispatchTool("call-1", 1, "book_flight", {"flight_id": "FL1", "passenger_name": "Ana"},
                         "book_flight:x#0", "call-1", 1, True)
        ex.submit(d)
        ex.submit(d)
        await asyncio.sleep(0.01)
        gate.set()
        await until(lambda: len(posted) == 2)
        ex.submit(d)
        await until(lambda: len(posted) == 3)
        assert backend.effects == 1 and ex.backend_invocations == 1
        assert all(r.ok and r.payload == {"booking_ref": "BR-1"} for r in posted)
        await ex.aclose()

    run(main())


def test_R5_transient_failure_is_retried_safely():
    """INVARIANT 8: definite no-effect failure → retry same operation → one effect."""

    class BookReasoner(CityReasoner):
        async def propose(self, req):
            from fdagent.runtime.reasoner import Draft

            done = req.snapshot["calls"] and all(c["status"] == "succeeded" for c in req.snapshot["calls"])
            return Draft((PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"}),),
                         reply="Booked." if done else None)

    async def main():
        backend, sink = GatedBackend(), RecordingSink()
        backend.fail_first.add("book_flight")
        rt = make_runtime(reasoner=BookReasoner(), backend=backend, sink=sink)
        await rt.start()
        say(rt, "Book FL1 for Ana")
        await until(lambda: "Booked." in sink.said)
        [op] = rt.kernel.state.ops.values()
        assert (op.attempt, op.status.value, backend.effects) == (2, "succeeded", 1)
        assert decisions(rt, "retry")[0]["idempotency_key"] == op.call_id
        await rt.aclose()

    run(main())


def test_R6_barge_in_stops_agent_speech():
    async def main():
        sink = RecordingSink(seconds=1.0)
        rt = make_runtime(sink=sink)
        await rt.start()
        say(rt, "hello there")
        await until(lambda: rt.kernel.state.agent_speaking)
        rt.post(UserSpeechStarted())
        await until(lambda: marks(rt, "speech_end"))
        assert marks(rt, "speech_end")[0]["interrupted"] is True
        assert decisions(rt, "barge_in")
        await rt.aclose()

    run(main())


def test_R6b_shutdown_mid_utterance_terminates():
    """Closing (or asyncio.run cleanup) while the agent is speaking must not hang."""

    async def main(clean: bool):
        rt = make_runtime(sink=RecordingSink(seconds=10.0))
        await rt.start()
        say(rt, "hello there")
        await until(lambda: rt.kernel.state.agent_speaking)
        if clean:
            await asyncio.wait_for(rt.aclose(), timeout=1.0)
        # else: return with tasks alive; asyncio.run must cancel them promptly

    t0 = time.monotonic()
    run(main(clean=True))
    run(main(clean=False))
    assert time.monotonic() - t0 < 3.0


def test_R7_latency_marks_are_recorded():
    async def main():
        sink = RecordingSink()
        rt = make_runtime(sink=sink)
        await rt.start()
        say(rt, "Find a flight to Oslo")
        await until(lambda: "Here are flights to Oslo." in sink.said)
        names = {r["name"] for r in rt.kernel.journal.records if r["kind"] == "note"}
        assert {"reasoning_start", "reasoning_end", "tool_start", "tool_end", "speech_start"} <= names
        await rt.aclose()

    run(main())
