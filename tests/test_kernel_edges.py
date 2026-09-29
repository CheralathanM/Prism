"""Edge cases for the fast path, stability gate, speech timing, and explicit failure."""

from __future__ import annotations

from fdagent.core.events import UserSpeechStarted, UserTranscript
from fdagent.core.model import desired_call_key

from .harness import Harness, ProposedCall as PC


def test_reply_is_held_until_turn_is_stable():
    h = Harness()
    h.user_says("Hi there, I need some help")
    assert not h.speech(h.propose([], reply="Sure, what do you need?"))
    speech = h.speech(h.stabilize())
    assert [(s.kind, s.text) for s in speech] == [("final", "Sure, what do you need?")]  # no extra backchannel


def test_backchannel_when_plan_not_ready_at_stability():
    h = Harness()
    h.user_says("Find flights to Oslo on June 2")
    assert [s.kind for s in h.speech(h.stabilize())] == ["backchannel"]


def test_filler_only_segment_does_not_replan():
    h = Harness()
    h.user_says("Find flights to Oslo on June 2")
    gen = h.s.generation
    n_req = len(h.decisions("reasoning_requested"))
    h.send(UserTranscript("um, uh"))
    assert h.s.generation == gen
    assert len(h.decisions("reasoning_requested")) == n_req


def test_stale_stability_timer_is_ignored():
    h = Harness()
    h.user_says("Find flights to Oslo")
    old_epoch = h.s.speech_epoch
    h.send(UserSpeechStarted())  # user resumes before the window closes
    from fdagent.core.events import TimerFired

    h.send(TimerFired(f"stability:{old_epoch}", "stability", epoch=old_epoch))
    assert not h.s.stable
    assert h.decisions("timer_stale")


def test_speech_pause_then_filler_still_gets_fresh_plan():
    """Speech onset bumps the generation; if the user only says 'um', stability must
    trigger a fresh plan instead of deadlocking on a stale one."""
    h = Harness()
    h.user_says("Find flights to Oslo on June 2")
    h.user_says("um")
    acts = h.stabilize()
    assert h.decisions("reasoning_requested")[-1]["cause"] == "stable_without_fresh_plan"
    [d] = h.dispatches(h.propose([PC("search_flights", {"destination": "Oslo", "date": "June 2"})]))
    assert d.args["destination"] == "Oslo"
    assert acts  # something was emitted (request and/or backchannel)


def test_invalid_args_fail_explicitly():
    h = Harness()
    h.user_says("Find flights to Oslo")
    h.stabilize()
    acts = h.propose([PC("search_flights", {"destination": "Oslo"})], reply="Here are flights to Oslo.")
    assert not h.dispatches(acts)
    assert "missing required argument 'date'" in h.decisions("invalid_call")[0]["error"]
    assert [s.kind for s in h.speech(acts)] == ["failure"]


def test_timeout_on_state_change_is_not_auto_retried():
    from fdagent.core.events import TimerFired

    h = Harness()
    h.user_says("Book FL1 for Ana")
    h.stabilize()
    [d] = h.dispatches(h.propose([PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"})]))
    acts = h.send(TimerFired(f"timeout:{d.call_id}:1", "tool_timeout", call_id=d.call_id, attempt=1))
    assert not h.dispatches(acts)
    assert h.op(d.call_id).status.value == "unknown"
    # a late confirmation resolves the uncertainty
    h.result(d.call_id, payload={"booking_ref": "BR-1"})
    assert h.op(d.call_id).status.value == "succeeded"
    assert len(h.dispatches()) == 1


def test_key_normalisation():
    assert desired_call_key("t", {"city": " New  York", "n": 2}) == desired_call_key("t", {"city": "new york", "n": 2.0})
    assert desired_call_key("t", {"city": "Lisbon"}) != desired_call_key("t", {"city": "Porto"})
    assert desired_call_key("t", {"a": 1}, occurrence=1) != desired_call_key("t", {"a": 1})


def test_user_returning_to_superseded_intent_revives_not_duplicates():
    h = Harness()
    book1 = PC("book_flight", {"flight_id": "FL1", "passenger_name": "Ana"})
    book2 = PC("book_flight", {"flight_id": "FL2", "passenger_name": "Ana"})
    h.user_says("Book FL1 for Ana")
    h.stabilize()
    [d1] = h.dispatches(h.propose([book1]))
    h.user_says("no, FL2")
    h.propose([book2])
    h.user_says("sorry, FL1 was right")
    h.propose([book1])
    h.stabilize()
    assert [d.args["flight_id"] for d in h.dispatches()] == ["FL1"]  # FL2 never ran, FL1 not re-created
    assert h.decisions("revived")
