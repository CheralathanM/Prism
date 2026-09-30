"""In-car tools (specs for the existing tool protocol) and a deterministic mock car backend."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any

from fdagent.core.model import ToolParam, ToolSpec

CAR_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("plan_route", "Plan a driving route to a destination and estimate the arrival time.",
             (ToolParam("destination", "string", True, "Where the driver wants to go, as they said it"),)),
    ToolSpec("start_navigation", "Start turn-by-turn navigation along a planned route.",
             (ToolParam("route_id", "string", True, "route_id returned by plan_route"),),
             state_changing=True),
)


def _slug(s: str) -> str:
    return "-".join(s.lower().split())


class MockCarBackend:
    """Deterministic in-process car system.

    ``plan_route`` is read-only (ETA derived from the destination text); ``start_navigation`` is a
    state change recorded in ``navigation_log`` so tests can prove it happened exactly once.
    ``slow_destinations`` adds latency to plan_route for chosen destinations (to let a correction
    arrive while the old route is still being planned).
    """

    def __init__(self, slow_destinations: dict[str, float] | None = None) -> None:
        self.slow = {k.lower(): v for k, v in (slow_destinations or {}).items()}
        self.calls: list[tuple[str, dict]] = []
        self.navigation_log: list[str] = []
        self._routes: dict[str, str] = {}
        self._lock = threading.Lock()

    def call(self, tool: str, args: dict[str, Any], idempotency_key: str) -> Any:
        with self._lock:
            self.calls.append((tool, dict(args)))
        if tool == "plan_route":
            dest = args["destination"]
            time.sleep(self.slow.get(dest.lower(), 0.0))
            eta = 8 + int(hashlib.sha1(dest.lower().encode()).hexdigest(), 16) % 25
            route_id = f"R-{_slug(dest)}"
            with self._lock:
                self._routes[route_id] = dest
            return {"status": "success", "route_id": route_id, "destination": dest, "eta_minutes": eta}
        if tool == "start_navigation":
            with self._lock:
                dest = self._routes.get(args["route_id"])
                if dest is None:
                    return {"status": "error", "message": f"unknown route {args['route_id']}"}
                self.navigation_log.append(dest)
            return {"status": "success", "navigating_to": dest, "route_id": args["route_id"]}
        return {"status": "error", "message": f"unknown tool {tool}"}
