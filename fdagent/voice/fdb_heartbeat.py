"""FDB-v3 latency telemetry, same semantics and line format as the official templates.

The harness reads ``LATENCY_TRACK_JSON:`` lines from /tmp/agent_heartbeat.log (only for
the optional ``search_latency_breakdown``). The template records user_done_at at the first
final transcript, agent_start_at at the next speaking start, and logs a breakdown only if a
tool started in between; then it resets.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_HEARTBEAT_LOG = "/tmp/agent_heartbeat.log"
_lock = threading.Lock()


class FdbLatencyLog:
    def __init__(self, room_name: str, path: str | None = None, clock=time.time) -> None:
        self.room = room_name
        self.path = Path(path or os.getenv("FDB_HEARTBEAT_LOG", DEFAULT_HEARTBEAT_LOG))
        self._clock = clock
        self._reset()

    def _reset(self) -> None:
        self.user_done_at = 0.0
        self.tool_start_at = 0.0
        self.tool_end_at = 0.0
        self.query_received = False

    def on_user_final(self, text: str) -> None:
        if not self.query_received:
            self.user_done_at = self._clock()
            self.query_received = True

    def on_mark(self, fields: dict[str, Any]) -> None:
        name = fields.get("name")
        if name == "tool_start":
            self.tool_start_at = fields.get("wall", self._clock())
        elif name == "tool_end":
            self.tool_end_at = fields.get("wall", self._clock())
        elif name == "speech_start" and self.query_received:
            self._log(fields.get("wall", self._clock()))
            self._reset()

    def _log(self, agent_start_at: float) -> None:
        if not (self.user_done_at and self.tool_start_at):
            return
        metrics = {
            "room": self.room,
            "tool": "Search Tool",
            "reasoning": round(self.tool_start_at - self.user_done_at, 3),
            "execution": round((self.tool_end_at - self.tool_start_at) if self.tool_end_at else 0, 3),
            "synthesis": round(agent_start_at - (self.tool_end_at or self.user_done_at), 3),
            "total": round(agent_start_at - self.user_done_at, 3),
            "agent_start_at": agent_start_at,
        }
        with _lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"LATENCY_TRACK_JSON: {json.dumps(metrics)}\n")
