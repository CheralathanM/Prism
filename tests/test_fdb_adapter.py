"""FDB-v3 adapter tests against the pinned, unmodified harness code (skipped if not fetched)."""

from __future__ import annotations

import ast
import asyncio
import json

import pytest

from fdagent.adapters.fdb_tools import DEFAULT_FDB_V3_DIR, FDB_TOOL_SPECS, FdbMockBackend
from fdagent.adapters.tool_protocol import to_openai_tool
from fdagent.core.events import UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.runtime.reasoner import Draft

from .fakes import RecordingSink, make_runtime, until
from .harness import ProposedCall as PC, ref

needs_fdb = pytest.mark.skipif(not (DEFAULT_FDB_V3_DIR / "mock_apis.py").exists(), reason="FDB-v3 not fetched")


@needs_fdb
def test_specs_match_official_template_signatures():
    """Function and argument names must match the harness exactly (the evaluator compares them)."""
    tree = ast.parse((DEFAULT_FDB_V3_DIR / "lk_agent_tool.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AssistantFnc")
    official = {}
    for fn in cls.body:
        if isinstance(fn, ast.AsyncFunctionDef):
            names = [a.arg for a in fn.args.args[1:]]
            n_defaults = len(fn.args.defaults)
            official[fn.name] = {n: i < len(names) - n_defaults for i, n in enumerate(names)}
    ours = {s.name: {p.name: p.required for p in s.params} for s in FDB_TOOL_SPECS}
    assert ours == official


def test_openai_schema_export():
    schema = to_openai_tool(next(s for s in FDB_TOOL_SPECS if s.name == "calculate_commute"))
    assert schema["function"]["parameters"]["required"] == ["origin_address", "destination_address"]


class ShopReasoner:
    async def propose(self, req):
        calls = (
            PC("search_products", {"query": "headphones", "max_price": 100}),
            PC("add_to_cart", {"product_id": ref(0, "products", 0, "product_id")}, depends_on=(0,)),
        )
        done = req.snapshot["calls"] and all(c["status"] == "succeeded" for c in req.snapshot["calls"])
        return Draft(calls, reply="Added the headphones to your cart." if done else None)


@needs_fdb
def test_two_step_chain_on_official_mocks_and_isolated_rooms(tmp_path):
    """INVARIANTS 11, 12: chained call on the real mock backend; two concurrent sessions
    write telemetry only under their own room and share no state."""
    log_path = tmp_path / "agent_tool_calls.log"

    async def session(room: str, sink: RecordingSink):
        rt = make_runtime(reasoner=ShopReasoner(), backend=FdbMockBackend(room, tool_log_path=str(log_path)),
                          sink=sink, tools=FDB_TOOL_SPECS, session_id=room)
        await rt.start()
        rt.post(UserSpeechStarted())
        rt.post(UserTranscript("Find me headphones under 100 dollars and add them to my cart"))
        rt.post(UserTurnEnded())
        await until(lambda: "Added the headphones to your cart." in sink.said)
        await rt.aclose()

    async def main():
        s1, s2 = RecordingSink(), RecordingSink()
        await asyncio.gather(session("eval-aaaa1111", s1), session("eval-bbbb2222", s2))

    asyncio.run(main())
    lines = [json.loads(l) for l in log_path.read_text().splitlines()]
    for room in ("eval-aaaa1111", "eval-bbbb2222"):
        calls = [l["call"] for l in lines if l["room"] == room]
        assert [c["function"] for c in calls] == ["search_products", "add_to_cart"]
        assert calls[0]["args"] == {"query": "headphones", "max_price": 100}
        assert calls[1]["args"] == {"quantity": 1, "product_id": "PROD1"}  # template default filled
        assert calls[0]["timestamp_start"] <= calls[0]["timestamp_end"] <= calls[1]["timestamp_start"]
