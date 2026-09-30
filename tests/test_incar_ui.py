"""Demo UI: the server starts, the scripted museum -> harbour interruption runs on the real
kernel/runtime, the streamed view shows the supersession, and a replay re-renders identically."""

from __future__ import annotations

import asyncio

from aiohttp.test_utils import TestClient, TestServer

from fdagent.extensions.incar.ui.server import DemoApp


def _strip(events):
    return [{k: v for k, v in e.items() if k != "seq"} for e in events if e["type"] not in ("reset", "replay")]


async def _scenario(tmp_path):
    demo = DemoApp(journal_dir=tmp_path, route_delay_s=1.5, speech_cps=20.0, interrupt_after_s=0.5)
    async with TestClient(TestServer(demo.make_app())) as client:
        page = await (await client.get("/")).text()
        for area in ("Conversation", "Agent state", "Full-duplex interruption", "Tool execution log", "Run demo"):
            assert area in page

        sse = await client.get("/api/events")
        assert sse.headers["Content-Type"].startswith("text/event-stream")
        sse.close()

        assert (await (await client.post("/api/run")).json())["ok"]
        for _ in range(600):
            status = await (await client.get("/api/status")).json()
            if status["busy"] is None:
                break
            await asyncio.sleep(0.05)
        assert status["busy"] is None
        live = await (await client.get("/api/history")).json()

        replayed = await (await client.post("/api/replay")).json()
        after = await (await client.get("/api/history")).json()
    return status, live, replayed, after


def test_ui_runs_interruption_demo_and_replays_identically(tmp_path):
    status, live, replayed, after = asyncio.run(_scenario(tmp_path))
    by = lambda t: [e for e in live if e["type"] == t]  # noqa: E731

    assert [e["text"] for e in by("user")] == ["Take me to the museum, please.", "Actually, go to the harbour instead."]
    steps = [e["step"] for e in by("interrupt")]
    for step in ("barge_in", "cancelled", "admitted", "superseded", "rejected"):
        assert step in steps, steps

    tools = by("tool")
    museum = [e["status"] for e in tools if e["label"] == 'plan_route(destination="museum")']
    assert "superseded" in museum and "late result rejected" in museum and "completed" not in museum
    nav = [e for e in tools if e["label"].startswith("start_navigation")]
    assert [e["status"] for e in nav if e["status"] == "dispatched"] == ["dispatched"]
    assert all(e["label"] == 'start_navigation(route_id="R-harbour")' for e in nav)
    assert [e["navigating_to"] for e in by("vehicle")] == ["harbour"]
    assert status["session"]["navigation_log"] == ["harbour"]

    assert any(e["text"].startswith("Navigating to the harbour") for e in by("agent"))
    assert any(e["status"] == "interrupted" for e in by("agent_status"))
    states = [e["state"] for e in by("state")]
    for s in ("Interrupted", "Cancelling previous action", "Executing tool", "Speaking", "Completed"):
        assert s in states, states

    # Journal replay: kernel decisions identical, and the view re-renders the same events.
    assert replayed["ok"] and replayed["identical"] and replayed["steps"] > 0
    assert _strip(after) == _strip(live)


async def _mic_flow(tmp_path):
    """Microphone path: speech onset is posted first, the final text later (started=True)."""
    demo = DemoApp(journal_dir=tmp_path, route_delay_s=1.5, speech_cps=10.0)
    async with TestClient(TestServer(demo.make_app())) as client:
        await client.post("/api/speech_start")
        await client.post("/api/say", json={"text": "Take me to the museum, please.", "started": True})
        for _ in range(200):
            if demo.session.kernel.state.agent_speaking:
                break
            await asyncio.sleep(0.02)
        await client.post("/api/speech_start")          # driver talks over the agent
        await asyncio.sleep(0.3)
        await client.post("/api/say", json={"text": "Actually, go to the harbour instead.", "started": True})
        for _ in range(400):
            if demo.session.view.state == "Completed" and not demo.session.view.inflight:
                break
            await asyncio.sleep(0.02)
        history = await (await client.get("/api/history")).json()
        return demo.session.backend.navigation_log, history


def test_ui_microphone_path_barges_in_and_navigates_once(tmp_path):
    nav, history = asyncio.run(_mic_flow(tmp_path))
    assert nav == ["harbour"]
    steps = [e["step"] for e in history if e["type"] == "interrupt"]
    assert "barge_in" in steps and "superseded" in steps
    assert [e["text"] for e in history if e["type"] == "user"] == [
        "Take me to the museum, please.", "Actually, go to the harbour instead."]


def test_scripted_planner_asks_again_when_destination_not_understood():
    from types import SimpleNamespace

    from fdagent.extensions.incar.demo import ScriptedCarPlanner

    conv = [{"role": "user", "text": "Take me to the museum"},
            {"role": "agent", "text": "Okay, planning a route to the museum.", "kind": "progress"},
            {"role": "user", "text": "No, take me to the thunder"}]          # misheard destination
    req = SimpleNamespace(snapshot={"conversation": conv, "calls": []})
    draft = asyncio.run(ScriptedCarPlanner().propose(req))
    assert draft.reply_kind == "clarify" and "didn't catch" in draft.reply
    assert [c.args for c in draft.calls][0] == {"destination": "museum"}   # plan kept, nothing cancelled
