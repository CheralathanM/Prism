"""The ten mandated interruption / tool-safety scenarios, each tied to an explicit invariant.

All tests are deterministic: events are fed one by one with a fake clock; no network,
no threads, no sleeps.
"""

from __future__ import annotations

from fdagent.core.actions import AdvisoryCancel, RequestReasoning
from fdagent.core.events import ToolResult, UserTranscript, UserTurnEnded
from fdagent.core.replay import replay

from .harness import Harness, OpStatus, ProposedCall as PC, ref

DELHI = {"destination": "Delhi", "date": "May 3"}
MUMBAI = {"destination": "Mumbai", "date": "May 3"}


def _delhi_then_mumbai():
    """Delhi search is in flight when the user self-corrects to Mumbai."""
    h = Harness()
    h.user_says("Find me a flight to Delhi on May 3")
    h.stabilize()
    [d1] = h.dispatches(h.propose([PC("search_flights", DELHI)]))
    h.user_says("Actually, Mumbai.")
    h.propose([PC("search_flights", MUMBAI)])
    [d2] = h.dispatches(h.stabilize())
    return h, d1, d2


def test_T1_argument_change_while_old_tool_running():
    """INVARIANT 5: a self-correction supersedes the old intent; only corrected args dispatch."""
    h, d1, d2 = _delhi_then_mumbai()
    assert d1.args["destination"] == "Delhi"
    assert d2.args["destination"] == "Mumbai"
    assert d2.call_id != d1.call_id and d2.key != d1.key
    assert h.op(d1.call_id).status == OpStatus.SUPERSEDED
    assert any(isinstance(a, AdvisoryCancel) and a.call_id == d1.call_id for a in h.log)
    assert any(d.get("correction_cue") == "actually" for d in h.decisions("generation_bump"))
    # Mumbai was held by the commit gate until the corrected turn was stable.
    assert {"key": d2.key, "tool": "search_flights", "reason": "intent_not_stable", "kind": "gate_blocked"} in h.decisions("gate_blocked")


def test_T2_stale_result_after_new_intent():
    """INVARIANT 6: a late result for a superseded call is logged but never mutates current state."""
    h, d1, d2 = _delhi_then_mumbai()
    acts = h.result(d1.call_id, payload={"flights": [{"flight_id": "FL-DEL"}]})
    assert acts == []  # no reasoning, no speech, no dispatch
    assert h.decisions("result_rejected")[-1]["reason"] == "superseded"
    op1 = h.op(d1.call_id)
    assert op1.status == OpStatus.SUPERSEDED and op1.result is None and op1.late_outcome["ok"]
    assert [c["args"]["destination"] for c in h.k.snapshot()["calls"]] == ["Mumbai"]

    acts = h.result(d2.call_id, payload={"flights": [{"flight_id": "FL-BOM"}]})
    assert any(isinstance(a, RequestReasoning) for a in acts)
    [spoken] = h.speech(h.propose([PC("search_flights", MUMBAI)], reply="Flight FL-BOM to Mumbai is available."))
    assert "Mumbai" in spoken.text and spoken.kind == "final"


def test_T3_same_state_changing_request_delivered_twice():
    """INVARIANT 7: the same state-changing request never executes twice."""
    h = Harness()
    book = PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"})
    h.user_says("Book FL1 for Ana")
    h.stabilize()
    h.propose([book])
    h.propose([book])  # identical proposal event delivered twice
    h.propose([book, book])  # duplicated inside one proposal
    h.user_says("Yes, book FL1 for Ana please")  # user repeats the request
    h.propose([book])
    h.stabilize()
    assert len(h.dispatches()) == 1
    assert h.decisions("proposal_duplicate_collapsed")


def test_T4_retry_after_partial_completion_is_idempotent():
    """INVARIANT 8: a retry after an ambiguous failure reuses the operation's identity,
    so an idempotent backend applies the effect exactly once."""

    class IdempotentBackend:
        def __init__(self):
            self.effects = 0
            self.seen: dict[str, dict] = {}

        def call(self, d):
            if d.idempotency_key not in self.seen:
                self.effects += 1
                self.seen[d.idempotency_key] = {"booking_ref": "BR-1"}
            return self.seen[d.idempotency_key]

    backend = IdempotentBackend()
    h = Harness()
    h.user_says("Book FL1 for Ana")
    h.stabilize()
    [d1] = h.dispatches(h.propose([PC("book_flight_idem", {"flight_id": "FL1", "passenger_name": "Ana"})]))
    backend.call(d1)  # effect committed, but the response is lost
    [d2] = h.dispatches(h.result(d1.call_id, ok=False, error="connection reset", retriable=True))
    assert (d2.call_id, d2.idempotency_key, d2.attempt) == (d1.call_id, d1.idempotency_key, 2)
    h.result(d2.call_id, payload=backend.call(d2), attempt=2)
    assert backend.effects == 1
    assert h.op(d1.call_id).status == OpStatus.SUCCEEDED
    # A straggler reply from attempt 1 is rejected.
    h.result(d1.call_id, payload={"booking_ref": "BR-1"}, attempt=1)
    assert h.decisions("result_rejected")[-1]["reason"] == "stale_attempt"


def test_T4b_non_idempotent_state_change_is_not_blindly_retried():
    """INVARIANT 8: unknown effect + non-idempotent backend → no retry, honest failure."""
    h = Harness()
    h.user_says("Book FL1 for Ana")
    h.stabilize()
    book = PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"})
    h.propose([book])
    [d1] = h.dispatches()
    acts = h.result(d1.call_id, ok=False, error="connection reset", retriable=True)
    assert not h.dispatches(acts)
    speech = h.speech(h.propose([book], reply="Your flight is booked!"))
    assert [s.kind for s in speech] == ["failure"]


def test_T5_three_step_chain():
    """INVARIANT 11: A → result A → B(args from A) → result B → C; completion only at the end."""
    h = Harness()
    chain = [
        PC("search_flights", DELHI),
        PC("book_flight", {"flight_id": ref(0, "flights", 0, "flight_id"), "passenger_name": "Ana"}, depends_on=(0,)),
        PC("send_confirmation", {"booking_ref": ref(1, "booking_ref")}, depends_on=(1,)),
    ]
    h.user_says("Book the first flight to Delhi on May 3 for Ana and send me the confirmation")
    h.stabilize()
    acts = h.propose(chain, reply="All done!")
    [d_search] = h.dispatches(acts)
    assert d_search.tool == "search_flights"
    assert not h.speech(acts) or all(s.kind == "backchannel" for s in h.speech(acts))  # no premature "done"

    [d_book] = h.dispatches(h.result(d_search.call_id, payload={"flights": [{"flight_id": "FL-DEL-7"}]}))
    assert d_book.tool == "book_flight" and d_book.args["flight_id"] == "FL-DEL-7"
    assert h.op(d_book.call_id).parent_keys == (d_search.key,)

    [d_conf] = h.dispatches(h.result(d_book.call_id, payload={"booking_ref": "BR-42"}))
    assert d_conf.args == {"booking_ref": "BR-42"}

    acts = h.result(d_conf.call_id, payload={"sent": True})
    assert any(isinstance(a, RequestReasoning) for a in acts)
    [spoken] = h.speech(h.propose(chain, reply="Booked FL-DEL-7, reference BR-42, confirmation sent."))
    assert spoken.kind == "final"
    assert [d.tool for d in h.dispatches()] == ["search_flights", "book_flight", "send_confirmation"]


def _chain(dest):
    return [
        PC("search_flights", {"destination": dest, "date": "May 3"}),
        PC("book_flight", {"flight_id": ref(0, "flights", 0, "flight_id"), "passenger_name": "Ana"}, depends_on=(0,)),
    ]


def test_T6_interruption_during_step_2_of_chain():
    """INVARIANTS 5, 6, 10, 11: interrupting mid-chain supersedes the in-flight step, the late
    effect is recorded as an orphan (never used), and the new chain runs cleanly."""
    h = Harness()
    h.user_says("Book a flight to Delhi on May 3 for Ana")
    h.stabilize()
    [d_s1] = h.dispatches(h.propose(_chain("Delhi")))
    [d_b1] = h.dispatches(h.result(d_s1.call_id, payload={"flights": [{"flight_id": "FL-DEL"}]}))

    h.user_says("Wait, make it Mumbai instead")
    acts = h.propose(_chain("Mumbai"))
    assert any(isinstance(a, AdvisoryCancel) and a.call_id == d_b1.call_id for a in acts)
    [d_s2] = h.dispatches(h.stabilize())
    assert d_s2.args["destination"] == "Mumbai"

    h.result(d_b1.call_id, payload={"booking_ref": "BR-DEL"})  # Delhi booking lands late
    assert h.decisions("result_rejected")[-1]["reason"] == "superseded"
    assert [e["args"]["flight_id"] for e in h.s.orphan_effects] == ["FL-DEL"]

    [d_b2] = h.dispatches(h.result(d_s2.call_id, payload={"flights": [{"flight_id": "FL-BOM"}]}))
    assert d_b2.args["flight_id"] == "FL-BOM"
    h.result(d_b2.call_id, payload={"booking_ref": "BR-BOM"})
    books = [d.args["flight_id"] for d in h.dispatches() if d.tool == "book_flight"]
    assert books == ["FL-DEL", "FL-BOM"]  # each exactly once; the stale one never re-dispatched
    assert h.k.snapshot()["orphan_effects"][0]["payload"] == {"booking_ref": "BR-DEL"}


def test_T7_many_events_while_slow_reasoner_runs():
    """INVARIANT 1/2: the kernel keeps processing events while reasoning is outstanding;
    a slow, outdated proposal cannot drive tools."""
    h = Harness()
    h.user_says("I need a flight to Rome", end_turn=False)
    req1 = h.latest_request()
    h.send(UserTranscript("no wait, Milan"))
    h.send(UserTranscript("on June 3"))
    h.send(UserTurnEnded())
    h.stabilize()
    req3 = h.latest_request()
    assert req3.request_id != req1.request_id
    assert req3.snapshot["transcript"][-2:] == ["no wait, Milan", "on June 3"]

    acts = h.propose([PC("search_flights", {"destination": "Rome", "date": "June 1"})], request=req1)
    assert not h.dispatches(acts)
    assert h.decisions("proposal_rejected")[-1]["reason"] == "superseded_request"

    [d] = h.dispatches(h.propose([PC("search_flights", {"destination": "Milan", "date": "June 3"})], request=req3))
    assert d.args["destination"] == "Milan"
    assert len(h.dispatches()) == 1


def test_T8_late_duplicate_result_cannot_overwrite_success():
    """INVARIANT 6: once an operation has a terminal outcome, later results for it are rejected."""
    h, d1, d2 = _delhi_then_mumbai()
    h.result(d2.call_id, payload={"flights": [{"flight_id": "FL-BOM"}]})
    h.result(d2.call_id, payload={"flights": [{"flight_id": "FL-WRONG"}]})
    assert h.decisions("result_rejected")[-1]["reason"] == "duplicate_result"
    assert h.op(d2.call_id).result == {"flights": [{"flight_id": "FL-BOM"}]}
    h.send(ToolResult("call-999", 1, True, {"x": 1}))
    assert h.decisions("result_rejected")[-1]["reason"] == "unknown_call"


def test_T9_tool_failure_blocks_completion_claim():
    """INVARIANTS 9, 10: a failed action is never reported as done."""
    h = Harness()
    h.user_says("Book FL1 for Ana")
    h.stabilize()
    book = PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"})
    [d] = h.dispatches(h.propose([book], reply="Your flight is booked!"))
    assert not [s for s in h.speech() if s.kind == "final"]  # withheld: call not yet succeeded
    h.result(d.call_id, ok=False, error="flight sold out")
    speech = h.speech(h.propose([book], reply="Your flight is booked!"))
    assert [s.kind for s in speech] == ["failure"]
    assert "sold out" in speech[0].text
    assert all("booked!" not in s.text for s in h.speech())


def test_T10_new_intent_makes_old_desired_action_obsolete():
    """INVARIANTS 5, 7: a planned-but-gated action from an old intent never executes."""
    h = Harness()
    h.user_says("Book FL1 for Ana", end_turn=False)  # user still talking
    assert not h.dispatches(h.propose([PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"})]))
    assert h.decisions("gate_blocked")[-1]["reason"] == "user_speaking"
    h.send(UserTranscript("actually make that FL2"))
    h.send(UserTurnEnded())
    h.stabilize()
    assert not h.dispatches()  # old plan is stale; waits for the fresh plan
    assert h.decisions("gate_blocked")[-1]["reason"] == "stale_plan"
    [d] = h.dispatches(h.propose([PC("book_flight", {"flight_id": "FL2", "passenger_name": "Ana"})]))
    assert d.args["flight_id"] == "FL2"
    assert [x.args["flight_id"] for x in h.dispatches()] == ["FL2"]


def test_journal_replay_is_deterministic():
    """INVARIANT 13: the journal alone reproduces every decision and action."""
    h, d1, d2 = _delhi_then_mumbai()
    h.result(d1.call_id, payload={"flights": []})
    h.result(d2.call_id, payload={"flights": [{"flight_id": "FL-BOM"}]})
    h.propose([PC("search_flights", MUMBAI)], reply="Found FL-BOM.")
    assert replay(h.journal.records) == []
