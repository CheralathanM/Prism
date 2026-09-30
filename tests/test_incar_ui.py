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
