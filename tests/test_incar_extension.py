"""In-car extension regression: a destination change during route planning (with barge-in) must
supersede the old route and start navigation exactly once, to the corrected destination."""

from __future__ import annotations

import asyncio

from fdagent.core.replay import replay
from fdagent.extensions.incar.demo import run_demo
from fdagent.extensions.incar.tools import CAR_TOOL_SPECS, MockCarBackend


def test_destination_change_supersedes_old_route_and_navigates_once():
    res = asyncio.run(run_demo(quiet=True))
    # Navigation (the only state change) happened exactly once, to the corrected destination.
    assert res.navigation_log == ["harbour"]
    assert [a for t, a in res.tool_calls if t == "start_navigation"] == [{"route_id": "R-harbour"}]
    # The stale museum route was planned but superseded; its late result never reached state.
    assert ("plan_route", {"destination": "museum"}) in res.tool_calls
    assert res.decisions.get("superseded", 0) >= 1
    assert res.decisions.get("result_rejected", 0) >= 1
    # The driver interrupted the agent mid-sentence.
    assert res.decisions.get("barge_in", 0) >= 1 and res.interrupted
    # Final answer is about the corrected destination only; nothing claims museum navigation.
    assert any(s.startswith("Navigating to the harbour") for s in res.said)
    assert not any("Navigating to the museum" in s for s in res.said)
    # The kernel journal replays deterministically.
    assert replay(res.journal_records) == []


def test_mock_car_backend_is_deterministic_and_validates_routes():
    b = MockCarBackend()
    r1 = b.call("plan_route", {"destination": "library"}, "k1")
    r2 = b.call("plan_route", {"destination": "library"}, "k2")
    assert r1 == r2 and r1["route_id"] == "R-library"
    assert b.call("start_navigation", {"route_id": "R-nowhere"}, "k3")["status"] == "error"
    assert b.call("start_navigation", {"route_id": "R-library"}, "k4")["navigating_to"] == "library"
    assert [s.name for s in CAR_TOOL_SPECS if s.state_changing] == ["start_navigation"]
