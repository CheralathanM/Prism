"""Session kernel: the single writer of session state.

``step(envelope)`` is synchronous and pure with respect to the outside world: it reads
one event, updates state, and returns actions. It never awaits a model or a tool, never
reads a clock, and never generates randomness, so a journal replays deterministically.

Step anatomy:
    handler(event)      → state update + immediate actions (fast path)
    reconcile()         → supersede / reuse / gate / dispatch
    _after_reconcile()  → completion-guarded speech, chain continuation
    journal.record()
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from .actions import Action, RequestReasoning, Speak, StartTimer, StopSpeaking
from .admission import admit_result
from .config import KernelConfig
from .events import (
    AgentSpeechEnded,
    AgentSpeechStarted,
    Event,
    ReasonerFailed,
    ReasonerProposal,
    TimerFired,
    ToolResult,
    UserSpeechStarted,
    UserTranscript,
    UserTurnEnded,
)
from .fast_path import correction_cue, is_filler_only
from .ids import IdGen
from .inbox import Envelope
from .journal import Journal
from .model import DesiredCall, OpStatus, ToolSpec, desired_call_key, is_ref
from .reconcile import reconcile, terminal_failure
from .state import PendingReply, SessionState


class SessionKernel:
    def __init__(
        self,
        session_id: str,
        tools: list[ToolSpec],
        config: KernelConfig | None = None,
        journal: Journal | None = None,
        **journal_meta: Any,
    ) -> None:
        self.config = config or KernelConfig()
        self.state = SessionState(session_id=session_id, tools={t.name: t for t in tools})
        self.ids = IdGen(session_id)
        self.journal = journal
        self._last_seq = 0
        self._decisions: list[dict[str, Any]] = []
        self._reply_candidate: PendingReply | None = None
        self._outcome_this_step = False
        self._handlers: dict[type[Event], Callable[[Any], list[Action]]] = {
            UserSpeechStarted: self._on_speech_started,
            UserTranscript: self._on_transcript,
            UserTurnEnded: self._on_turn_ended,
            AgentSpeechStarted: self._on_agent_speech_started,
            AgentSpeechEnded: self._on_agent_speech_ended,
            ReasonerProposal: self._on_proposal,
            ReasonerFailed: self._on_reasoner_failed,
            ToolResult: self._on_tool_result,
            TimerFired: self._on_timer,
        }
        if journal is not None:
            journal.header(session_id, self.config, tools, **journal_meta)

    # ── public ──────────────────────────────────────────────────────────────
    def step(self, env: Envelope) -> list[Action]:
        if env.seq <= self._last_seq:
            raise ValueError(f"inbox order violated: seq {env.seq} after {self._last_seq}")
        self._last_seq = env.seq
        self._decisions = []
        self._reply_candidate = None
        self._outcome_this_step = False

        actions = self._handlers[type(env.event)](env.event)
        actions += reconcile(self.state, self.config, self.ids, self._note)
        actions += self._after_reconcile()

        if self.journal is not None:
            self.journal.record(env, self._decisions, actions)
        return actions

    def snapshot(self) -> dict[str, Any]:
        """Read-only view for workers (deep copy; mutating it changes nothing)."""
        s = self.state
        calls = []
        for dc in s.desired_in_order():
            op = s.op_for_key(dc.key)
            calls.append(
                {
                    "key": dc.key,
                    "tool": dc.tool,
                    "args": op.args if op else dc.args,
                    "status": op.status.value if op else ("invalid" if dc.key in s.invalid else "planned"),
                    "result": op.result if op else None,
                    "error": (op.error if op else s.invalid.get(dc.key)),
                }
            )
        return copy.deepcopy(
            {
                "session_id": s.session_id,
                "generation": s.generation,
                "transcript": [t["text"] for t in s.transcript],
                "calls": calls,
                "orphan_effects": s.orphan_effects,
                "tools": sorted(s.tools),
            }
        )

    # ── helpers ─────────────────────────────────────────────────────────────
    def _note(self, kind: str, **fields: Any) -> None:
        self._decisions.append({"kind": kind, **fields})

    def _request_reasoning(self, cause: str) -> RequestReasoning:
        s = self.state
        rid = self.ids.next("req")
        s.latest_request_id = rid
        s.latest_request_generation = s.generation
        self._note("reasoning_requested", request_id=rid, generation=s.generation, cause=cause)
        return RequestReasoning(rid, s.generation, self.snapshot())

    def _ensure_reasoning(self, cause: str) -> list[Action]:
        s = self.state
        if s.transcript and s.latest_request_generation != s.generation:
            return [self._request_reasoning(cause)]
        return []

    def _bump_generation(self, cause: str, **fields: Any) -> None:
        self.state.generation += 1
        self._note("generation_bump", generation=self.state.generation, cause=cause, **fields)

    # ── user / perception ───────────────────────────────────────────────────
    def _on_speech_started(self, ev: UserSpeechStarted) -> list[Action]:
        s = self.state
        s.user_speaking = True
        s.turn_open = True
        s.stable = False
        s.speech_epoch += 1
        s.pending_reply = None
        self._bump_generation("speech_started")
        if s.agent_speaking:
            self._note("barge_in")
            return [StopSpeaking("barge_in")]
        return []

    def _on_transcript(self, ev: UserTranscript) -> list[Action]:
        s = self.state
        if not ev.final:
            s.partial = ev.text
            return []
        s.partial = ""
        text = ev.text.strip()
        if not text:
            return []
        if is_filler_only(text):
            # Fillers carry no intent; do not churn generations or re-plan.
            self._note("filler_ignored", text=text)
            return []
        cue = correction_cue(text)
        self._bump_generation("transcript", correction_cue=cue)
        s.transcript.append({"generation": s.generation, "text": text})
        return [self._request_reasoning("transcript")]

    def _on_turn_ended(self, ev: UserTurnEnded) -> list[Action]:
        s = self.state
        s.user_speaking = False
        return [
            StartTimer(
                timer_id=f"stability:{s.speech_epoch}",
                kind="stability",
                delay_s=self.config.stability_s,
                generation=s.generation,
                epoch=s.speech_epoch,
            )
        ]

    def _on_agent_speech_started(self, ev: AgentSpeechStarted) -> list[Action]:
        self.state.agent_speaking = True
        return []

    def _on_agent_speech_ended(self, ev: AgentSpeechEnded) -> list[Action]:
        self.state.agent_speaking = False
        return []

    # ── timers ──────────────────────────────────────────────────────────────
    def _on_timer(self, ev: TimerFired) -> list[Action]:
        s = self.state
        if ev.kind == "stability":
            if ev.epoch != s.speech_epoch or s.user_speaking:
                self._note("timer_stale", timer_id=ev.timer_id)
                return []
            s.stable = True
            s.turn_open = False
            self._note("intent_stable", generation=s.generation, epoch=s.speech_epoch)
            actions = self._ensure_reasoning("stable_without_fresh_plan")
            if s.pending_reply and s.pending_reply.generation == s.generation:
                self._reply_candidate = s.pending_reply
            s.pending_reply = None
            if (
                self._reply_candidate is None
                and self.config.backchannel
                and s.transcript
                and s.acked_generation != s.generation
                and s.replied_generation != s.generation
            ):
                s.acked_generation = s.generation
                actions.append(Speak(self.config.backchannel_text, "backchannel", s.generation))
                self._note("backchannel", generation=s.generation)
            return actions

        if ev.kind == "tool_timeout":
            op = s.ops.get(ev.call_id or "")
            if op and op.status == OpStatus.DISPATCHED and op.attempt == ev.attempt:
                op.status = OpStatus.UNKNOWN
                self._note("tool_timeout", call_id=op.call_id, key=op.key, attempt=op.attempt)
                self._outcome_this_step = True
            return []

        self._note("timer_unknown_kind", kind=ev.kind)
        return []

    # ── reasoner ────────────────────────────────────────────────────────────
    def _on_proposal(self, ev: ReasonerProposal) -> list[Action]:
        s = self.state
        if ev.request_id != s.latest_request_id:
            self._note("proposal_rejected", request_id=ev.request_id, reason="superseded_request",
                       latest=s.latest_request_id)
            return []
        if ev.generation != s.generation:
            self._note("proposal_rejected", request_id=ev.request_id, reason="stale_generation",
                       proposal_generation=ev.generation, generation=s.generation)
            return []

        desired: dict[str, DesiredCall] = {}
        keys: list[str | None] = []
        for i, c in enumerate(ev.calls):
            bad_dep = [j for j in c.depends_on if not (0 <= j < i) or keys[j] is None]
            refs = [v["$ref"][0] for v in c.args.values() if is_ref(v)]
            if bad_dep or any(r not in c.depends_on for r in refs):
                keys.append(None)
                self._note("proposal_call_dropped", index=i, tool=c.tool, reason="bad_dependency")
                continue
            args = {
                k: ({"$ref": [keys[v["$ref"][0]], *v["$ref"][1:]]} if is_ref(v) else v)
                for k, v in c.args.items()
            }
            key = desired_call_key(c.tool, args, c.occurrence)
            keys.append(key)
            if key in desired:
                self._note("proposal_duplicate_collapsed", index=i, key=key)
                continue
            deps = tuple(k for j in c.depends_on if (k := keys[j]) is not None)
            desired[key] = DesiredCall(key, c.tool, args, deps, ev.generation, i)

        before = list(s.desired)
        s.desired = desired
        s.desired_generation = ev.generation
        s.invalid = {k: v for k, v in s.invalid.items() if k in desired}
        self._note(
            "plan_accepted",
            request_id=ev.request_id,
            generation=ev.generation,
            keys=list(desired),
            added=[k for k in desired if k not in before],
            removed=[k for k in before if k not in desired],
        )

        if ev.reply:
            reply = PendingReply(ev.request_id, ev.generation, ev.reply, ev.reply_kind)
            if s.stable and not s.user_speaking:
                self._reply_candidate = reply
            else:
                s.pending_reply = reply
                self._note("reply_held_until_stable", request_id=ev.request_id)
        return []

    def _on_reasoner_failed(self, ev: ReasonerFailed) -> list[Action]:
        s = self.state
        if ev.request_id != s.latest_request_id or ev.generation != s.generation:
            self._note("reasoner_failure_ignored", request_id=ev.request_id)
            return []
        self._note("reasoner_failed", request_id=ev.request_id, error=ev.error)
        s.replied_generation = s.generation
        return [Speak(self.config.fallback_text, "fallback", s.generation)]

    # ── tools ───────────────────────────────────────────────────────────────
    def _on_tool_result(self, ev: ToolResult) -> list[Action]:
        adm = admit_result(self.state, ev)
        fields = {"call_id": ev.call_id, "attempt": ev.attempt, "ok": ev.ok,
                  "key": adm.op.key if adm.op else None}
        if adm.admitted:
            self._note("result_admitted", **fields)
            self._outcome_this_step = True
        else:
            self._note("result_rejected", reason=adm.reason, **fields)
        return []

    # ── post-reconcile: speech + chain continuation ────────────────────────
    def _after_reconcile(self) -> list[Action]:
        s = self.state
        actions: list[Action] = []
        if self._reply_candidate is not None:
            actions += self._speak_reply(self._reply_candidate)
        if self._outcome_this_step and not s.in_flight() and not s.user_speaking:
            # Every desired call is resolved or blocked: let the planner derive the next
            # chain step (A → result A → B) or the final answer from admitted facts.
            actions.append(self._request_reasoning("outcome_admitted"))
        return actions

    def _speak_reply(self, reply: PendingReply) -> list[Action]:
        """Completion guard: no 'done' without an admitted success for every desired call."""
        s = self.state
        if reply.request_id in s.spoken_requests:
            return []
        failures = {k: r for k in s.desired if (r := terminal_failure(s, k, self.config))}
        if failures:
            out: list[Action] = []
            for key, reason in failures.items():
                if key in s.announced_failures:
                    continue
                s.announced_failures.add(key)
                tool = s.desired[key].tool.replace("_", " ")
                out.append(Speak(f"Sorry, I couldn't complete the {tool} request: {reason}.", "failure", s.generation))
            self._note("completion_blocked", request_id=reply.request_id, failed=list(failures))
            s.spoken_requests.add(reply.request_id)
            s.replied_generation = s.generation
            return out
        if reply.kind == "final":
            pending = [k for k in s.desired if (op := s.op_for_key(k)) is None or op.status != OpStatus.SUCCEEDED]
            if pending:
                self._note("reply_withheld", request_id=reply.request_id, reason="calls_not_succeeded", pending=pending)
                return []
        s.spoken_requests.add(reply.request_id)
        s.replied_generation = s.generation
        self._note("speak", request_id=reply.request_id, reply_kind=reply.kind)
        return [Speak(reply.text, reply.kind, s.generation)]
