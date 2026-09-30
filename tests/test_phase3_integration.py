"""Phase 3 integration with fake providers: kernel → OpenAI reasoner (fake client) → tools,
plus the LiveKit ingress/egress adapters against fake LiveKit objects.

All vocabulary is synthetic and checked not to occur in the benchmark data.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from fdagent.adapters.fdb_tools import DEFAULT_FDB_V3_DIR, FDB_TOOL_SPECS, FdbMockBackend
from fdagent.core.config import KernelConfig
from fdagent.core.events import AgentSpeechEnded, UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.providers.openai_reasoner import OpenAIReasoner
from fdagent.runtime.loop import SessionRuntime
from fdagent.voice.ingress import IngressBridge
from fdagent.voice.speech_sink import LiveKitSpeechSink

from .fakes import RecordingSink, decisions, marks, until
from .test_reasoner_parsing import TOOLS_X


# ── fakes ───────────────────────────────────────────────────────────────────
class FakeOpenAI:
    """Mimics ``AsyncOpenAI().chat.completions.create``; a planner function maps the
    payload the reasoner sends to the JSON plan the 'model' returns."""

    def __init__(self, planner: Callable[[dict], Any], delay: Callable[[dict], float] = lambda p: 0.0):
        self.planner, self.delay, self.requests = planner, delay, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.requests.append(kw)
        payload = json.loads(kw["messages"][-1]["content"])
        await asyncio.sleep(self.delay(payload))
        out = self.planner(payload)
        content = out if isinstance(out, str) else json.dumps(out)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class RoomBackend:
    """Async backend for TOOLS_X with per-town gates and an effect counter."""

    def __init__(self):
        self.gates: dict[str, asyncio.Event] = {}
        self.calls: list[tuple[str, dict]] = []

    async def acall(self, tool, args, idem):
        self.calls.append((tool, args))
        if args.get("town") in self.gates:
            await self.gates[args["town"]].wait()
        if tool == "find_rooms":
            return {"rooms": [{"room_id": f"R-{args['town'][:1]}{args['beds']}"}]}
        return {"status": "success", "echo": args}


def user_texts(p):
    return [c["text"] for c in p["conversation"] if c["role"] == "user"]


def town_planner(p):
    """Latest mentioned town wins; reply once the search succeeded."""
    towns = [w for t in user_texts(p) for w in ("Eastvale", "Westbrook", "Northfield") if w in t]
    if not towns:
        return {"keep": [], "new_calls": [], "reply": "Which town?", "reply_kind": "clarify"}
    town = towns[-1]
    for c in p["calls"]:
        if c["args"].get("town") == town and c["status"] == "succeeded":
            return {"keep": [c["index"]], "new_calls": [], "reply": f"Found {c['result']['rooms'][0]['room_id']} in {town}.",
                    "reply_kind": "final"}
    return {"keep": [], "new_calls": [{"tool": "find_rooms", "args": {"town": town, "beds": "2", "budget": "$900"}}],
            "reply": None}


def build(planner, backend=None, tools=TOOLS_X, delay=lambda p: 0.0, sink=None, session="p3"):
    client = FakeOpenAI(planner, delay)
    kernel = SessionKernel(session, list(tools), KernelConfig(stability_s=0.05), journal=Journal())
    rt = SessionRuntime(kernel, OpenAIReasoner(tools, client=client, model="fake-model"),
                        backend or RoomBackend(), sink or RecordingSink())
    return rt, client


def say(rt, text, end=True):
    rt.post(UserSpeechStarted())
    rt.post(UserTranscript(text))
    if end:
        rt.post(UserTurnEnded())


def run(coro):
    return asyncio.run(coro)


# ── end-to-end: kernel → reasoner → tool ────────────────────────────────────
def test_E1_end_to_end_with_typed_args():
    async def main():
        sink, backend = RecordingSink(), RoomBackend()
        rt, client = build(town_planner, backend, sink=sink)
        await rt.start()
        say(rt, "um, rooms in Eastvale please")
        await until(lambda: any(s.startswith("Found") for s in sink.said))
        assert backend.calls == [("find_rooms", {"town": "Eastvale", "beds": 2, "budget": 900.0})]
        assert "Found R-E2 in Eastvale." in sink.said
        kw = client.requests[0]
        assert kw["model"] == "fake-model" and kw["temperature"] == 0 and kw["response_format"] == {"type": "json_object"}
        await rt.aclose()

    run(main())


def test_E2_correction_supersedes_in_flight_tool_call():
    async def main():
        sink, backend = RecordingSink(), RoomBackend()
        gate = backend.gates.setdefault("Eastvale", asyncio.Event())
        rt, _ = build(town_planner, backend, sink=sink)
        await rt.start()
        say(rt, "rooms in Eastvale")
        await until(lambda: backend.calls)
        say(rt, "actually, Westbrook")
        await until(lambda: "Found R-W2 in Westbrook." in sink.said)
        gate.set()
        await rt.idle()
        assert [a["town"] for _, a in backend.calls] == ["Eastvale", "Westbrook"]
        assert [d["reason"] for d in decisions(rt, "result_rejected")] == ["superseded"]
        assert not any("Eastvale" in s for s in sink.said)
        await rt.aclose()

    run(main())


def test_E3_invalid_arguments_never_reach_the_tool():
    def planner(p):
        return {"keep": [], "new_calls": [{"tool": "find_rooms", "args": {"town": "Eastvale", "beds": "a couple",
                                                                          "budget": 100}}], "reply": "Done!"}

    async def main():
        sink, backend = RecordingSink(), RoomBackend()
        rt, _ = build(planner, backend, sink=sink)
        await rt.start()
        say(rt, "rooms in Eastvale")
        await until(lambda: any("couldn't complete" in s for s in sink.said))
        assert backend.calls == [] and "Done!" not in sink.said
        assert decisions(rt, "invalid_call")[0]["error"] == "'beds' must be a number"
        await rt.aclose()

    run(main())


def test_E4_stale_and_cancelled_reasoning_cannot_drive_tools():
    slow_first = {"n": 0}

    def delay(p):
        slow_first["n"] += 1
        return 0.4 if slow_first["n"] == 1 else 0.0

    async def main():
        backend = RoomBackend()
        rt, client = build(town_planner, backend, delay=delay)
        await rt.start()
        say(rt, "rooms in Eastvale", end=False)
        await asyncio.sleep(0.02)
        rt.post(UserTranscript("no, Northfield"))
        rt.post(UserTurnEnded())
        await until(lambda: backend.calls)
        await asyncio.sleep(0.5)  # well past the first request's delay
        await rt.idle()
        assert [a["town"] for _, a in backend.calls] == ["Northfield"]
        assert marks(rt, "reasoning_cancelled")
        # the cancelled request never produced a proposal event
        assert all(d["request_id"] != "req-1" for d in decisions(rt, "plan_accepted"))
        await rt.aclose()

    run(main())


def test_E5_malformed_model_output_gives_honest_fallback():
    async def main():
        sink = RecordingSink()
        rt, _ = build(lambda p: "I think you want rooms", sink=sink)
        await rt.start()
        say(rt, "rooms in Eastvale")
        await until(lambda: sink.said)
        await rt.idle()
        assert decisions(rt, "reasoner_failed") and "PlanParseError" in decisions(rt, "reasoner_failed")[0]["error"]
        assert any("didn't catch" in s for s in sink.said)
        await rt.aclose()

    run(main())


@pytest.mark.skipif(not (DEFAULT_FDB_V3_DIR / "mock_apis.py").exists(), reason="FDB-v3 not fetched")
def test_E6_chain_on_official_mocks_via_reasoner(tmp_path):
    """Two-step chain where step 2's argument comes from step 1's result, planned
    iteratively by the reasoner (no $refs), executed on the unmodified FDB mocks."""

    def planner(p):
        calls = p["calls"]
        if not calls:
            return {"keep": [], "new_calls": [{"tool": "search_products", "args": {"query": "lantern kettle"}}],
                    "reply": "Searching now.", "reply_kind": "progress"}
        if len(calls) == 1 and calls[0]["status"] == "succeeded":
            pid = calls[0]["result"]["products"][0]["product_id"]
            return {"keep": [0], "new_calls": [{"tool": "add_to_cart", "args": {"product_id": pid, "quantity": "1"}}]}
        if all(c["status"] == "succeeded" for c in calls):
            return {"keep": [0, 1], "new_calls": [], "reply": "Added it to your cart.", "reply_kind": "final"}
        return {"keep": list(range(len(calls))), "new_calls": []}

    log = tmp_path / "calls.log"

    async def main():
        sink = RecordingSink()
        rt, _ = build(planner, FdbMockBackend("eval-test0001", tool_log_path=str(log)), tools=FDB_TOOL_SPECS, sink=sink)
        await rt.start()
        say(rt, "find a lantern kettle and add it to my cart")
        await until(lambda: "Added it to your cart." in sink.said)
        await rt.aclose()
        return sink.said

    said = run(main())
    assert said.index("Searching now.") < said.index("Added it to your cart.")
    calls = [json.loads(l)["call"] for l in log.read_text().splitlines()]
    assert [(c["function"], c["args"]) for c in calls] == [
        ("search_products", {"max_price": None, "query": "lantern kettle"}),
        ("add_to_cart", {"quantity": 1, "product_id": "PROD1"}),
    ]


# ── LiveKit adapters (fake LiveKit objects) ─────────────────────────────────
def test_L1_ingress_bridge_translates_livekit_events():
    posted = []
    b = IngressBridge(posted.append)
    b.on_user_state("speaking")
    b.on_user_state("speaking")  # duplicate state change is ignored
    b.on_transcript("uh", is_final=False)
    b.on_transcript("  ", is_final=True)  # empty final is normalized to "" but still forwarded
    b.on_transcript("rooms in Eastvale", is_final=True)
    b.on_user_state("listening")
    b.on_user_state("away")  # not speaking → no second turn end
    assert [type(e).__name__ for e in posted] == ["UserSpeechStarted", "UserTranscript", "UserTranscript",
                                                  "UserTranscript", "UserTurnEnded"]
    assert (posted[1].final, posted[2].text, posted[2].final, posted[3].text) == (False, "", True, "rooms in Eastvale")


class FakeHandle:
    def __init__(self, seconds, lk_interrupts=False):
        self.seconds, self.lk_interrupts, self.interrupted, self.interrupt_calls = seconds, lk_interrupts, False, 0

    def interrupt(self, force=False):
        self.interrupt_calls += 1
        self.interrupted = True

    async def wait_for_playout(self):
        await asyncio.sleep(self.seconds)
        if self.lk_interrupts:
            self.interrupted = True


class FakeSession:
    def __init__(self, **handle_kw):
        self.handles, self.handle_kw = [], handle_kw

    def say(self, text, allow_interruptions=True, add_to_chat_ctx=True):
        assert add_to_chat_ctx is False  # the kernel, not LiveKit's chat context, owns conversation state
        h = FakeHandle(**self.handle_kw)
        self.handles.append(h)
        return h


def test_L2_kernel_barge_in_interrupts_livekit_playout():
    async def main():
        session = FakeSession(seconds=5.0)
        rt, _ = build(lambda p: {"keep": [], "new_calls": [], "reply": "Hello there!", "reply_kind": "final"},
                      sink=LiveKitSpeechSink(session))
        await rt.start()
        say(rt, "hi")
        await until(lambda: rt.kernel.state.agent_speaking)
        t0 = time.monotonic()
        rt.post(UserSpeechStarted())
        await until(lambda: marks(rt, "speech_end"))
        assert time.monotonic() - t0 < 0.5
        assert session.handles[0].interrupt_calls == 1
        assert marks(rt, "speech_end")[0]["interrupted"] is True
        await rt.aclose()

    run(main())


def test_L3_livekit_side_interruption_is_reported():
    async def main():
        posted = []
        from fdagent.runtime.speech import SpeechChannel
        from fdagent.core.actions import Speak

        ch = SpeechChannel(LiveKitSpeechSink(FakeSession(seconds=0.01, lk_interrupts=True)), posted.append, lambda **k: None)
        ch.enqueue(Speak("Hello", "final", 1))
        await until(lambda: any(isinstance(e, AgentSpeechEnded) for e in posted))
        assert [e for e in posted if isinstance(e, AgentSpeechEnded)][0].interrupted is True
        await ch.aclose()

    run(main())


def test_L4_livekit_agent_module_imports_without_connecting():
    pytest.importorskip("livekit.agents")
    import fdagent.voice.livekit_agent as m

    assert m.server is not None and callable(m.entrypoint)
    cfg = m.AgentSettings.from_env({})
    assert cfg.provider == "fdagent" and cfg.latency_profile == "instant"
