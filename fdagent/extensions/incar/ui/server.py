"""Demo UI server: a thin visualization/control layer around the real in-car session.

    python -m fdagent.extensions.incar.ui                  # http://127.0.0.1:8765
    python -m fdagent.extensions.incar.ui --planner gemini # live planner (GOOGLE_API_KEY, server side only)

Every session uses the real ``SessionKernel`` + ``SessionRuntime`` (tool executor, speech channel,
timers) with the in-car tool specs, the mock car backend and the scripted planner from
``fdagent.extensions.incar``. The browser never runs any agent logic: the server streams the
kernel journal, translated by ``JournalView``, over Server-Sent Events. Binds to localhost only;
no credentials are ever sent to the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from aiohttp import web

from fdagent.core.config import KernelConfig
from fdagent.core.events import UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.core.replay import replay
from fdagent.runtime.loop import SessionRuntime

from ..demo import ScriptedCarPlanner
from ..tools import CAR_TOOL_SPECS, MockCarBackend
from .view import JournalView

STATIC = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JOURNAL_DIR = REPO_ROOT / "results" / "ui_journals"
FIRST_REQUEST = "Take me to the museum, please."
CORRECTION = "Actually, go to the harbour instead."


class StreamingJournal(Journal):
    """The standard journal, plus a callback per record (used to stream it to the browser)."""

    def __init__(self, path: Path, on_record) -> None:
        super().__init__(path)
        self._on_record = on_record

    def _write(self, rec: dict[str, Any]) -> None:
        super()._write(rec)
        self._on_record(self.records[-1])


class UISpeech:
    """Speech sink with simulated playback time (the browser voices the text locally)."""

    def __init__(self, chars_per_s: float) -> None:
        self.cps = chars_per_s

    async def say(self, text: str) -> bool:
        await asyncio.sleep(len(text) / self.cps)
        return False


class Hub:
    """Broadcasts UI events to all connected browsers; keeps history for late joiners."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.queues: set[asyncio.Queue] = set()

    def emit(self, ev: dict[str, Any]) -> None:
        ev = {"seq": len(self.history) + 1, **ev}
        self.history.append(ev)
        for q in self.queues:
            q.put_nowait(ev)

    def clear(self, mode: str) -> None:
        self.history = []
        self.emit({"type": "reset", "mode": mode})


class DemoSession:
    def __init__(self, hub: Hub, journal_dir: Path, planner_name: str, route_delay_s: float,
                 speech_cps: float, n: int) -> None:
        self.hub, self.planner_name = hub, planner_name
        self.session_id = f"incar-ui-{n}"
        self.journal_path = journal_dir / f"{self.session_id}.jsonl"
        self.view = JournalView()
        self.backend = MockCarBackend(slow_destinations={"museum": route_delay_s})
        cfg = KernelConfig(stability_s=0.3, backchannel=False)
        self.journal = StreamingJournal(self.journal_path, self._on_record)
        self.kernel = SessionKernel(self.session_id, list(CAR_TOOL_SPECS), cfg, journal=self.journal)
        self.runtime = SessionRuntime(self.kernel, make_planner(planner_name), self.backend, UISpeech(speech_cps))
        self.running = False

    def _on_record(self, rec: dict[str, Any]) -> None:
        for ev in self.view.feed(rec):
            self.hub.emit(ev)

    async def start(self) -> None:
        await self.runtime.start()
        self.running = True

    async def close(self) -> None:
        if self.running:
            self.running = False
            await self.runtime.aclose()
            self.journal.close()

    def speech_started(self) -> None:
        self.runtime.post(UserSpeechStarted())

    def say(self, text: str, started: bool = False) -> None:
        """One driver utterance. ``started`` = the microphone already reported speech onset."""
        if not started:
            self.runtime.post(UserSpeechStarted())
        self.runtime.post(UserTranscript(text))
        self.runtime.post(UserTurnEnded())

    def status(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "planner": self.planner_name, "running": self.running,
                "generation": self.view.generation, "state": self.view.state,
                "journal": str(self.journal_path.relative_to(REPO_ROOT)) if self.journal_path.is_relative_to(REPO_ROOT)
                else str(self.journal_path),
                "records": len(self.journal.records), "navigation_log": list(self.backend.navigation_log),
                "tool_calls": [[t, a] for t, a in self.backend.calls]}


def make_planner(name: str):
    if name == "gemini":
        from fdagent.providers.gemini_reasoner import GeminiReasoner
        return GeminiReasoner(CAR_TOOL_SPECS)
    return ScriptedCarPlanner()


async def _until(pred, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


class DemoApp:
    def __init__(self, journal_dir: Path = DEFAULT_JOURNAL_DIR, planner: str = "scripted",
                 route_delay_s: float = 8.0, speech_cps: float = 13.0, interrupt_after_s: float = 1.2) -> None:
        self.hub = Hub()
        self.journal_dir, self.planner = Path(journal_dir), planner
        self.route_delay_s, self.speech_cps, self.interrupt_after_s = route_delay_s, speech_cps, interrupt_after_s
        self.session: DemoSession | None = None
        self.last_journal: Path | None = None
        self.n = 0
        self.busy: str | None = None       # "demo" | "replay" while a scripted run is in progress
        self._task: asyncio.Task | None = None

    async def new_session(self) -> DemoSession:
        if self.session is not None:
            await self.session.close()
            self.last_journal = self.session.journal_path
        self.n += 1
        self.hub.clear("live")
        self.session = DemoSession(self.hub, self.journal_dir, self.planner, self.route_delay_s, self.speech_cps, self.n)
        await self.session.start()
        return self.session

    async def run_demo(self) -> None:
        """Scripted flow: museum request -> interrupt while the agent speaks -> harbour."""
        s = await self.new_session()
        s.say(FIRST_REQUEST)
        await _until(lambda: any(t == "plan_route" for t, _ in s.backend.calls) and s.kernel.state.agent_speaking, 20)
        await asyncio.sleep(self.interrupt_after_s)
        s.say(CORRECTION)
        await _until(lambda: s.view.state == "Completed" and not s.kernel.state.agent_speaking, 40)
        # Let the stale route's late result arrive so the rejection is visible.
        await _until(lambda: not s.view.inflight, self.route_delay_s + 5)
        await s.runtime.idle(timeout_s=5)

    async def replay_last(self) -> dict[str, Any]:
        path = self.last_journal
        if self.session is not None and self.session.journal.records:
            await self.session.close()
            path = self.session.journal_path
            self.last_journal = path
            self.session = None
        if path is None or not path.exists():
            return {"ok": False, "error": "no journal to replay yet"}
        records = Journal.load(path)
        mismatches = replay(records)
        self.hub.clear("replay")
        self.hub.emit({"type": "replay", "phase": "start", "journal": path.name})
        view, prev = JournalView(), None
        for rec in records:
            t = rec.get("ts", rec.get("t"))
            if t is not None and prev is not None:
                await asyncio.sleep(min(max(t - prev, 0.0), 1.5))
            prev = t if t is not None else prev
            for ev in view.feed(rec):
                self.hub.emit(ev)
        result = {"ok": True, "journal": path.name, "steps": sum(r["kind"] == "step" for r in records),
                  "identical": not mismatches, "mismatches": len(mismatches)}
        self.hub.emit({"type": "replay", "phase": "done", **result})
        return result

    def _spawn(self, name: str, coro) -> None:
        async def wrapper():
            try:
                await coro
            except Exception as e:  # surface, never crash the server
                self.hub.emit({"type": "error", "text": f"{name} failed: {e!r}"})
            finally:
                self.busy = None
        self.busy = name
        self._task = asyncio.create_task(wrapper())

    # -- HTTP --------------------------------------------------------------------------
    async def index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "index.html")

    async def events(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(self.hub.history)
        self.hub.queues.add(q)
        try:
            for ev in backlog:
                await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": keepalive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.hub.queues.discard(q)
        return resp

    async def history(self, request: web.Request) -> web.Response:
        return web.json_response(self.hub.history)

    async def status(self, request: web.Request) -> web.Response:
        return web.json_response({"busy": self.busy, "planner": self.planner,
                                  "last_journal": self.last_journal.name if self.last_journal else None,
                                  "session": self.session.status() if self.session else None})

    async def run(self, request: web.Request) -> web.Response:
        if self.busy:
            return web.json_response({"ok": False, "error": f"{self.busy} in progress"}, status=409)
        self._spawn("demo", self.run_demo())
        return web.json_response({"ok": True})

    async def say(self, request: web.Request) -> web.Response:
        if self.busy:
            return web.json_response({"ok": False, "error": f"{self.busy} in progress"}, status=409)
        body = await request.json()
        text = str(body.get("text", "")).strip()[:300]
        if not text:
            return web.json_response({"ok": False, "error": "empty"}, status=400)
        if self.session is None or not self.session.running:
            await self.new_session()
        self.session.say(text, started=bool(body.get("started")))
        return web.json_response({"ok": True})

    async def speech_start(self, request: web.Request) -> web.Response:
        """Microphone heard the driver start talking (lets a spoken interruption barge in at once)."""
        if self.busy:
            return web.json_response({"ok": False, "error": f"{self.busy} in progress"}, status=409)
        if self.session is None or not self.session.running:
            await self.new_session()
        self.session.speech_started()
        return web.json_response({"ok": True})

    async def reset(self, request: web.Request) -> web.Response:
        if self.busy == "demo" and self._task:
            self._task.cancel()
        if self.busy == "replay":
            return web.json_response({"ok": False, "error": "replay in progress"}, status=409)
        await self.new_session()
        return web.json_response({"ok": True, "session": self.session.status()})

    async def replay(self, request: web.Request) -> web.Response:
        if self.busy:
            return web.json_response({"ok": False, "error": f"{self.busy} in progress"}, status=409)
        self.busy = "replay"
        try:
            return web.json_response(await self.replay_last())
        finally:
            self.busy = None

    async def on_cleanup(self, app: web.Application) -> None:
        if self._task:
            self._task.cancel()
        if self.session:
            await self.session.close()

    def make_app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.get("/", self.index), web.get("/api/events", self.events), web.get("/api/history", self.history),
            web.get("/api/status", self.status), web.post("/api/run", self.run), web.post("/api/say", self.say),
            web.post("/api/speech_start", self.speech_start),
            web.post("/api/reset", self.reset), web.post("/api/replay", self.replay),
        ])
        app.on_cleanup.append(self.on_cleanup)
        return app


def main() -> None:
    ap = argparse.ArgumentParser(description="In-car full-duplex demo UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--planner", choices=("scripted", "gemini"), default="scripted")
    ap.add_argument("--route-delay", type=float, default=8.0, help="mock latency of the museum route (s)")
    ap.add_argument("--journal-dir", default=str(DEFAULT_JOURNAL_DIR))
    args = ap.parse_args()
    if args.planner == "gemini":
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env.local")
        os.environ.pop("OPENAI_API_KEY", None)
    demo = DemoApp(Path(args.journal_dir), args.planner, args.route_delay)
    print(f"In-car demo UI on http://{args.host}:{args.port}  (planner={args.planner})")
    web.run_app(demo.make_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
