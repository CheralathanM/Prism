"""Regression tests from the six-case validation run.

1. Pending transcription (split-request pattern): the user's request was split over several
   speech segments; the gate dispatched on the first transcript after 0.6 s of silence while
   the segment carrying the price limit was still being transcribed. Dispatch must wait until
   every ended segment has its transcript (or has been explicitly closed by a deadline).
2. Stale queued speech (late-correction pattern): a clarification planned before the user's
   correction arrived was still played afterwards. Queued speech from an older generation
   must be discarded when a newer transcript arrives.
Synthetic values only.
"""

from __future__ import annotations

import asyncio

from fdagent.core.actions import DiscardStaleSpeech, Speak
from fdagent.core.events import TimerFired, UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.core.model import ToolParam, ToolSpec
from fdagent.runtime.speech import SpeechChannel

from .fakes import RecordingSink, until
from .harness import TOOLS, Harness, ProposedCall as PC

FIND = ToolSpec("find_items", "Find items", (ToolParam("query", "string"),
                                             ToolParam("max_price", "number", required=False)))


def segment(h: Harness, text: str | None = None):
    """One VAD speech segment: onset, (optional transcript arriving early), end of speech."""
    h.send(UserSpeechStarted())
    if text is not None:
        h.send(UserTranscript(text))
    h.send(UserTurnEnded())


# ── pending transcription ───────────────────────────────────────────────────
def test_argument_in_a_later_segment_is_not_lost_by_premature_dispatch():
    """Split-request pattern: "...a standing lamp" / "...under 80 dollars" as two segments whose
    transcripts arrive late; silence alone must not release the call without the price."""
    h = Harness(tools=TOOLS + [FIND])
    segment(h)  # "um, I'm looking for a standing lamp" (still transcribing)
    segment(h)  # "something under 80 dollars" (still transcribing)
    assert h.s.pending_transcripts == [1, 2]
    h.send(UserTranscript("um, I'm looking for a standing lamp"))  # first transcript lands
    h.stabilize()  # the user has been silent long enough...
    acts = h.propose([PC("find_items", {"query": "standing lamp"})])
    assert not h.dispatches(acts)  # ...but the second segment is still in the STT pipeline
    assert h.decisions("gate_blocked")[-1]["reason"] == "transcript_pending"

    h.send(UserTranscript("something under 80 dollars"))  # settles the last segment; new intent
    [d] = h.dispatches(h.propose([PC("find_items", {"query": "standing lamp", "max_price": 80})]))
    assert d.args == {"query": "standing lamp", "max_price": 80}
    assert len(h.dispatches()) == 1  # the price-less call never ran


def test_a_segment_without_transcript_is_closed_by_its_deadline():
    h = Harness(tools=TOOLS + [FIND])
    segment(h, "find a standing lamp")  # transcript arrived first: nothing pending
    assert h.s.pending_transcripts == []
    segment(h)  # noise segment whose transcript never arrives
    assert h.s.pending_transcripts == [2]
    h.stabilize()  # requests a fresh plan for the current generation
    assert not h.dispatches(h.propose([PC("find_items", {"query": "standing lamp"})]))
    acts = h.send(TimerFired("transcript:2", "transcript_deadline", epoch=2), dt=10.0)
    assert [d.args for d in h.dispatches(acts)] == [{"query": "standing lamp"}]
    assert h.decisions("transcript_timeout")[0]["epoch"] == 2


def test_empty_and_filler_transcripts_settle_their_segments():
    h = Harness(tools=TOOLS + [FIND])
    segment(h)
    segment(h)
    h.send(UserTranscript(""))  # STT produced nothing for the first segment
    h.send(UserTranscript("um, uh"))  # filler only for the second
    assert h.s.pending_transcripts == []
    assert h.decisions("filler_ignored")


def test_late_timer_for_an_already_settled_segment_is_harmless():
    h = Harness(tools=TOOLS + [FIND])
    segment(h)
    h.send(UserTranscript("find a standing lamp"))
    h.send(TimerFired("transcript:1", "transcript_deadline", epoch=1), dt=10.0)
    assert h.s.pending_transcripts == [] and not h.decisions("transcript_timeout")


def test_supersession_still_wins_while_transcripts_are_pending():
    """A correction in a later segment replaces the earlier value; only the final call runs."""
    h = Harness()
    segment(h)  # "flights to Oslo on May 3"
    segment(h)  # "no wait, make that Bergen"
    h.send(UserTranscript("flights to Oslo on May 3"))
    h.stabilize()
    assert not h.dispatches(h.propose([PC("search_flights", {"destination": "Oslo", "date": "May 3"})]))
    h.send(UserTranscript("no wait, make that Bergen"))
    [d] = h.dispatches(h.propose([PC("search_flights", {"destination": "Bergen", "date": "May 3"})]))
    assert d.args["destination"] == "Bergen"
    assert [x.args["destination"] for x in h.dispatches()] == ["Bergen"]


def test_journal_replay_still_deterministic_with_transcript_gating():
    from fdagent.core.replay import replay

    h = Harness(tools=TOOLS + [FIND])
    segment(h)
    segment(h)
    h.send(UserTranscript("I'm looking for a standing lamp"))
    h.stabilize()
    h.propose([PC("find_items", {"query": "standing lamp"})])
    h.send(TimerFired("transcript:2", "transcript_deadline", epoch=2), dt=10.0)
    assert replay(h.journal.records) == []


# ── stale queued speech ─────────────────────────────────────────────────────
def test_new_transcript_tells_the_speech_channel_to_drop_older_generations():
    h = Harness()
    h.user_says("Find flights to Oslo on May 3")
    acts = h.send(UserTranscript("actually, make it May 4"))
    [dsc] = [a for a in acts if isinstance(a, DiscardStaleSpeech)]
    assert dsc.below_generation == h.s.generation
    assert not [a for a in h.send(UserTranscript("um")) if isinstance(a, DiscardStaleSpeech)]  # fillers don't


def test_stale_clarification_queued_behind_an_acknowledgement_is_not_played():
    """Late-correction pattern: the acknowledgement is playing, a clarification for the same (old)
    generation is queued behind it, then the user's correction arrives."""

    async def main():
        marks, posted = [], []
        sink = RecordingSink(seconds=0.2)
        ch = SpeechChannel(sink, posted.append, lambda **k: marks.append(k))
        ch.enqueue(Speak("Okay, one moment.", "backchannel", 3))
        ch.enqueue(Speak("What would you like to do?", "clarify", 3))
        await until(lambda: sink.said == ["Okay, one moment."])
        ch.discard_below(4)  # a newer transcript (generation 4) arrived
        ch.enqueue(Speak("Late speech for the old intent", "progress", 3))  # arrives after the floor moved
        ch.enqueue(Speak("Here is the updated answer.", "final", 4))
        await until(lambda: "Here is the updated answer." in sink.said)
        await ch.aclose()
        return sink.said, marks

    said, marks = asyncio.run(main())
    assert said == ["Okay, one moment.", "Here is the updated answer."]  # playing ack finished normally
    dropped = [m["text"] for m in marks if m["name"] == "speech_dropped_stale"]
    assert dropped == ["What would you like to do?", "Late speech for the old intent"]
