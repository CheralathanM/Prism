"""The heartbeat log must match the official template's LATENCY_TRACK_JSON semantics."""

from __future__ import annotations

import json

from fdagent.voice.fdb_heartbeat import FdbLatencyLog


def test_breakdown_logged_only_when_a_tool_ran(tmp_path):
    path = tmp_path / "hb.log"
    hb = FdbLatencyLog("eval-x", path=str(path), clock=lambda: 100.0)
    hb.on_user_final("hello")
    hb.on_mark({"name": "speech_start", "wall": 101.0})  # small-talk reply, no tool → nothing logged
    assert not path.exists()

    hb.on_user_final("do the thing")  # re-armed after reset
    hb.on_mark({"name": "tool_start", "wall": 100.4})
    hb.on_mark({"name": "tool_end", "wall": 100.9})
    hb.on_mark({"name": "speech_start", "wall": 101.5})
    [line] = path.read_text().splitlines()
    assert line.startswith("LATENCY_TRACK_JSON: ")
    m = json.loads(line[len("LATENCY_TRACK_JSON: "):])
    assert m == {"room": "eval-x", "tool": "Search Tool", "reasoning": 0.4, "execution": 0.5, "synthesis": 0.6,
                 "total": 1.5, "agent_start_at": 101.5}
