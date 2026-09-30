"""In-car destination-change demo on the real kernel/runtime (no LiveKit needed).

    python -m fdagent.extensions.incar.demo                     # deterministic scripted planner
    python -m fdagent.extensions.incar.demo --planner gemini    # live Gemini planner (GOOGLE_API_KEY)
    python -m fdagent.extensions.incar.demo --speak-dir out/    # also render each utterance with Piper

Scenario: "Take me to the museum" -> the agent plans a route (slow) and starts saying so -> the
driver barges in: "Actually, go to the harbour instead." Expected: the agent stops talking,
the museum route is superseded (its late result is rejected), navigation is never started to the
museum, and navigation starts exactly once to the harbour, followed by the final answer.
Every event, decision and action is written to the kernel journal.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fdagent.core.config import KernelConfig
from fdagent.core.events import ProposedCall, UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.runtime.loop import SessionRuntime
from fdagent.runtime.reasoner import Draft

from .tools import CAR_TOOL_SPECS, MockCarBackend

PLACES = ("harbour", "museum", "library", "beach", "convention centre")


class ScriptedCarPlanner:
    """Deterministic stand-in for the LLM planner (same Draft contract): the destination most
    recently mentioned by the driver wins; route first, then navigation using its route_id."""

    async def propose(self, request) -> Draft:
        await asyncio.sleep(0.05)
        snap = request.snapshot
        said = " ".join(c["text"] for c in snap["conversation"] if c["role"] == "user").lower()
        hits = [(said.rfind(p), p) for p in PLACES if p in said]
        if not hits:
            return Draft(reply="Where would you like to go?", reply_kind="clarify")
        dest = max(hits)[1]
        calls = (ProposedCall("plan_route", {"destination": dest}),
                 ProposedCall("start_navigation", {"route_id": {"$ref": [0, "route_id"]}}, depends_on=(0,)))
        last = snap["conversation"][-1] if snap["conversation"] else None
        if last and last["role"] == "user" and not any(p in last["text"].lower() for p in PLACES):
            # The driver just said something without a destination we know (e.g. a misheard name):
            # keep the current plan untouched and ask, rather than silently repeating the old one.
            return Draft(calls, reply="Sorry, I didn't catch a destination. I can go to the "
                         + ", ".join(PLACES[:-1]) + " or " + PLACES[-1] + ".", reply_kind="clarify")
        state = {c["tool"]: c for c in snap["calls"]}
        nav = state.get("start_navigation")
        if nav and nav["status"] == "succeeded" and nav["result"].get("navigating_to") == dest:
            eta = state["plan_route"]["result"]["eta_minutes"]
            return Draft(calls, reply=f"Navigating to the {dest}. You'll arrive in about {eta} minutes.")
        if not snap["calls"] or all(c["status"] == "planned" for c in snap["calls"]):
            return Draft(calls, reply=f"Okay, planning a route to the {dest}.", reply_kind="progress")
        return Draft(calls)


class ConsoleSpeech:
    """Prints what the agent says; playback time is simulated (~15 chars/s). Optionally renders
    each utterance to a WAV file with the local Piper voice."""

    def __init__(self, t0: float, speak_dir: str | None = None, chars_per_s: float = 15.0) -> None:
        self.t0, self.cps, self.said, self.interrupted = t0, chars_per_s, [], []
        self.speak_dir = Path(speak_dir) if speak_dir else None
        self._piper = None
        if self.speak_dir:
            from fdagent.providers.local_piper_tts import PiperTTS
            self.speak_dir.mkdir(parents=True, exist_ok=True)
            self._piper = PiperTTS()

    async def say(self, text: str) -> bool:
        print(f"  [{time.monotonic() - self.t0:5.2f}s] AGENT says: {text}")
        self.said.append(text)
        if self._piper is not None:
            audio = await self._piper.synthesize(text)
            path = self.speak_dir / f"{len(self.said):02d}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(audio.sample_rate)
                w.writeframes(audio.data)
        try:
            await asyncio.sleep(len(text) / self.cps)
        except asyncio.CancelledError:
            print(f"  [{time.monotonic() - self.t0:5.2f}s] AGENT interrupted (barge-in)")
            self.interrupted.append(text)
            raise
        return False


@dataclass
class DemoResult:
    said: list[str]
    interrupted: list[str]
    tool_calls: list[tuple[str, dict]]
    navigation_log: list[str]
    decisions: dict[str, int] = field(default_factory=dict)
    journal_records: list[dict[str, Any]] = field(default_factory=list)


async def run_demo(planner: Any = None, journal_path: str | None = None, speak_dir: str | None = None,
                   quiet: bool = False) -> DemoResult:
    t0 = time.monotonic()
    log = (lambda *a: None) if quiet else print
    backend = MockCarBackend(slow_destinations={"museum": 2.0})
    speech = ConsoleSpeech(t0, speak_dir)
    kernel = SessionKernel("incar-demo", list(CAR_TOOL_SPECS), KernelConfig(stability_s=0.3, backchannel=False),
                           journal=Journal(journal_path))
    rt = SessionRuntime(kernel, planner or ScriptedCarPlanner(), backend, speech)
    await rt.start()

    def user(text: str) -> None:
        log(f"  [{time.monotonic() - t0:5.2f}s] DRIVER says: {text}")
        rt.post(UserSpeechStarted())
        rt.post(UserTranscript(text))
        rt.post(UserTurnEnded())

    async def until(pred, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            await asyncio.sleep(0.02)
        return False

    try:
        user("Take me to the museum, please.")
        # Wait until the (slow) museum route is being planned and the agent has started talking.
        await until(lambda: any(t == "plan_route" for t, _ in backend.calls) and kernel.state.agent_speaking, 15)
        await asyncio.sleep(0.6)
        user("Actually, go to the harbour instead.")  # barge-in + destination change
        await until(lambda: bool(backend.navigation_log) and not kernel.state.agent_speaking
                    and any("harbour" in s and "Navigating" in s for s in speech.said), 30)
        await asyncio.sleep(0.3)
        await rt.idle(timeout_s=10)
    finally:
        records = list(kernel.journal.records)
        await rt.aclose()

    from collections import Counter
    decisions = Counter(d["kind"] for r in records if r["kind"] == "step" for d in r["decisions"])
    return DemoResult(speech.said, speech.interrupted, backend.calls, backend.navigation_log,
                      dict(decisions), records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--planner", choices=("scripted", "gemini"), default="scripted")
    ap.add_argument("--journal", default=None, help="write the kernel journal (JSONL) here")
    ap.add_argument("--speak-dir", default=None, help="render each utterance to WAV with local Piper")
    args = ap.parse_args()
    planner = None
    if args.planner == "gemini":
        from dotenv import load_dotenv

        from fdagent.providers.gemini_reasoner import GeminiReasoner
        load_dotenv(Path(__file__).resolve().parents[3] / ".env.local")
        os.environ.pop("OPENAI_API_KEY", None)
        planner = GeminiReasoner(CAR_TOOL_SPECS)
    print(f"In-car destination-change demo (planner={args.planner})")
    res = asyncio.run(run_demo(planner, args.journal, args.speak_dir))
    rej = res.decisions.get("result_rejected", 0)
    print("\nSummary")
    print(f"  tool calls executed : {[(t, a) for t, a in res.tool_calls]}")
    print(f"  navigation started  : {res.navigation_log}  (exactly once, corrected destination only)")
    print(f"  superseded ops      : {res.decisions.get('superseded', 0)}   stale results rejected: {rej}")
    print(f"  barge-ins           : {res.decisions.get('barge_in', 0)}   interrupted speech: {res.interrupted}")
    if args.journal:
        print(f"  journal             : {args.journal}  (replay: python -m fdagent.core.replay {args.journal})")


if __name__ == "__main__":
    main()
