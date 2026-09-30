"""Journal records -> UI events. Pure and deterministic (a view, not a second kernel).

The kernel journal already contains everything the demo needs to show: user transcripts,
Speak actions, barge-in, supersession, result admission/rejection, dispatches, and the runtime's
speech/tool/reasoning marks (journal notes). ``JournalView.feed`` turns each record into zero or
more small UI events; replaying a stored journal through a fresh view gives the same events.
"""

from __future__ import annotations

from typing import Any

# Agent-state card values (display strings).
LISTENING, PLANNING, SPEAKING = "Listening", "Planning", "Speaking"
INTERRUPTED, CANCELLING, EXECUTING, COMPLETED = "Interrupted", "Cancelling previous action", "Executing tool", "Completed"


def fmt_call(tool: str, args: dict[str, Any]) -> str:
    inner = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items())
    return f"{tool}({inner})"


class JournalView:
    def __init__(self) -> None:
        self.t0: float | None = None
        self.calls: dict[str, dict[str, Any]] = {}      # call_id -> {tool, args, status}
        self.superseded: set[str] = set()
        self.inflight: set[str] = set()
        self.planning = 0
        self.speaking: dict[str, Any] | None = None     # current utterance (msg id, text, kind)
        self.msg_seq = 0
        self.pending_msgs: list[dict[str, Any]] = []     # Speak actions not yet started
        self.state = LISTENING
        self.generation = 0
        self.interrupted_turn = False
        self.barged = False                              # barge-in seen; speech end not yet reported
        self.routes: dict[str, dict[str, Any]] = {}      # route_id -> plan_route payload
        self.navigation: list[dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------------------------
    def _t(self, rec: dict[str, Any]) -> float:
        t = rec.get("ts", rec.get("t"))
        if t is None:
            return 0.0
        if self.t0 is None:
            self.t0 = t
        return round(t - self.t0, 3)

    def _state(self, out: list[dict], t: float, state: str | None = None) -> None:
        if state is None:
            # Derived from what is actually in progress; when nothing is, keep the last state
            # (only a new utterance returns the card to Listening).
            live = self.inflight - self.superseded
            speaking = self.speaking is not None and not self.barged
            state = SPEAKING if speaking else PLANNING if self.planning else EXECUTING if live else None
            if state is None:
                return
        if state != self.state:
            self.state = state
            out.append({"type": "state", "t": t, "state": state})

    # -- main ------------------------------------------------------------------------------
    def feed(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        kind = rec.get("kind")
        if kind == "header":
            return [{"type": "session", "session_id": rec["session_id"], "tools": [x["name"] for x in rec["tools"]]}]
        t = self._t(rec)
        out: list[dict[str, Any]] = []
        if kind == "note":
            self._note(rec, t, out)
        elif kind == "step":
            self._step(rec, t, out)
        return out

    def _note(self, n: dict[str, Any], t: float, out: list[dict]) -> None:
        name = n.get("name")
        if name == "reasoning_start":
            self.planning += 1
            self._state(out, t)
        elif name == "reasoning_end":
            self.planning = max(0, self.planning - 1)
            self._state(out, t)
        elif name == "tool_start":
            self.inflight.add(n["call_id"])
            self._state(out, t)
        elif name == "tool_end":
            self.inflight.discard(n["call_id"])
            self._state(out, t)
        elif name == "speech_start":
            msg = next((m for m in self.pending_msgs if m["text"] == n.get("text")), None)
            if msg is not None:
                self.pending_msgs.remove(msg)
            else:
                self.msg_seq += 1
                msg = {"id": self.msg_seq, "text": n.get("text", "")}
            self.speaking = msg
            out.append({"type": "agent_status", "t": t, "id": msg["id"], "status": "speaking", "text": msg["text"]})
            self._state(out, t)
        elif name == "speech_end":
            msg, self.speaking, self.barged = self.speaking, None, False
            if msg is not None:
                status = "interrupted" if n.get("interrupted") else "spoken"
                out.append({"type": "agent_status", "t": t, "id": msg["id"], "status": status, "text": msg["text"]})
                if n.get("interrupted"):
                    out.append({"type": "interrupt", "t": t, "step": "cancelled",
                                "text": f'Previous response cancelled mid-sentence: "{msg["text"]}"'})
            if not n.get("interrupted") and n.get("speech_kind") == "final":
                self._state(out, t, COMPLETED)
            else:
                self._state(out, t)
        elif name == "tool_cancel_advisory":
            c = self.calls.get(n["call_id"], {})
            out.append({"type": "tool", "t": t, "call_id": n["call_id"], "status": "cancel requested",
                        "label": fmt_call(c.get("tool", "?"), c.get("args", {}))})

    def _step(self, r: dict[str, Any], t: float, out: list[dict]) -> None:
        ev = r["event"]
        et = ev.get("type")
        if et == "UserSpeechStarted":
            self._state(out, t, LISTENING)
        elif et == "UserTranscript" and ev.get("final"):
            out.append({"type": "user", "t": t, "text": ev["text"]})
        for d in r["decisions"]:
            k = d.get("kind")
            if k == "generation_bump":
                self.generation = d["generation"]
                out.append({"type": "generation", "t": t, "generation": d["generation"]})
                if d.get("cause") == "transcript" and self.interrupted_turn:
                    cue = f' (correction cue: "{d["correction_cue"]}")' if d.get("correction_cue") else ""
                    out.append({"type": "interrupt", "t": t, "step": "admitted",
                                "text": f"New user intent admitted as generation {d['generation']}{cue}"})
                    self.interrupted_turn = False
            elif k == "barge_in":
                self.interrupted_turn = True
                self.barged = True
                out.append({"type": "interrupt", "t": t, "step": "barge_in",
                            "text": "Driver spoke while the agent was talking: speech stopped immediately"})
                self._state(out, t, INTERRUPTED)
            elif k == "dispatch":
                self.calls[d["call_id"]] = {"tool": d["tool"], "args": d["args"]}
                out.append({"type": "tool", "t": t, "call_id": d["call_id"], "status": "dispatched",
                            "label": fmt_call(d["tool"], d["args"]), "state_changing": d.get("state_changing", False)})
            elif k == "superseded":
                c = self.calls.get(d.get("call_id") or "", {"tool": d.get("tool", "?"), "args": {}})
                if d.get("call_id"):
                    self.superseded.add(d["call_id"])
                label = fmt_call(c["tool"], c["args"])
                out.append({"type": "tool", "t": t, "call_id": d.get("call_id"), "status": "superseded", "label": label})
                out.append({"type": "interrupt", "t": t, "step": "superseded",
                            "text": f"Old action superseded: {label} is no longer wanted"})
                self._state(out, t, CANCELLING)
            elif k == "result_rejected":
                c = self.calls.get(d["call_id"], {"tool": "?", "args": {}})
                label = fmt_call(c["tool"], c["args"])
                out.append({"type": "tool", "t": t, "call_id": d["call_id"], "status": "late result rejected",
                            "label": label, "reason": d.get("reason")})
                out.append({"type": "interrupt", "t": t, "step": "rejected",
                            "text": f"Late result of {label} arrived and was rejected ({d.get('reason')}): never applied"})
            elif k == "result_admitted":
                c = self.calls.get(d["call_id"], {"tool": "?", "args": {}})
                payload = ev.get("payload") if ev.get("call_id") == d["call_id"] else None
                status = "completed" if d.get("ok") else "failed"
                if c["tool"] == "start_navigation" and d.get("ok"):
                    status = "navigation started"
                out.append({"type": "tool", "t": t, "call_id": d["call_id"], "status": status,
                            "label": fmt_call(c["tool"], c["args"]), "result": payload})
                self._vehicle(c["tool"], payload, t, out)
            elif k == "gate_blocked" and d.get("reason") in ("waiting_dependency", "intent_not_stable"):
                out.append({"type": "gate", "t": t, "tool": d["tool"], "reason": d["reason"]})
        for a in r["actions"]:
            if a.get("type") == "Speak":
                self.msg_seq += 1
                msg = {"id": self.msg_seq, "text": a["text"]}
                self.pending_msgs.append(msg)
                out.append({"type": "agent", "t": t, "id": msg["id"], "text": a["text"], "kind": a.get("kind")})

    def _vehicle(self, tool: str, payload: dict | None, t: float, out: list[dict]) -> None:
        if not isinstance(payload, dict):
            return
        if tool == "plan_route" and payload.get("route_id"):
            self.routes[payload["route_id"]] = payload
        elif tool == "start_navigation":
            route = self.routes.get(payload.get("route_id", ""), {})
            self.navigation.append({"destination": payload.get("navigating_to"), "eta_minutes": route.get("eta_minutes")})
            out.append({"type": "vehicle", "t": t, "navigating_to": payload.get("navigating_to"),
                        "eta_minutes": route.get("eta_minutes"), "navigation_starts": len(self.navigation)})


def view_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v = JournalView()
    return [e for r in records for e in v.feed(r)]
