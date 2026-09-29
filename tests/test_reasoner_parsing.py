"""OpenAI reasoner: pure parsing / coercion / prompt-building tests (no network).

Tool vocabulary here is synthetic (see TOOLS_X) so no benchmark values appear in tests.
"""

from __future__ import annotations

import json

import pytest

from fdagent.core.model import ToolParam, ToolSpec
from fdagent.providers.openai_reasoner import PlanParseError, build_messages, coerce_args, parse_plan

TOOLS_X = (
    ToolSpec("find_rooms", "Find rooms", (ToolParam("town", "string"), ToolParam("beds", "integer"),
                                           ToolParam("budget", "number"))),
    ToolSpec("set_pref", "Set a preference", (ToolParam("name", "string"), ToolParam("value", "string")),
             state_changing=True),
    ToolSpec("reserve", "Reserve", (ToolParam("room_id", "string"), ToolParam("nights", "integer", required=False)),
             state_changing=True),
)
SPEC = {t.name: t for t in TOOLS_X}


def snap(calls=(), conversation=()):
    return {"generation": 3, "conversation": list(conversation), "calls": list(calls), "orphan_effects": [],
            "transcript": [c["text"] for c in conversation if c["role"] == "user"]}


# ── coercion ────────────────────────────────────────────────────────────────
def test_coerce_numbers_from_spoken_strings():
    args, warn = coerce_args(SPEC["find_rooms"], {"town": " Northfield ", "beds": "2", "budget": "$1,250"})
    assert args == {"town": "Northfield", "beds": 2, "budget": 1250.0} and warn == []


def test_coerce_integral_float_to_int_and_number_to_string():
    assert coerce_args(SPEC["find_rooms"], {"town": "A", "beds": 3.0, "budget": 900})[0]["beds"] == 3
    assert coerce_args(SPEC["set_pref"], {"name": "limit", "value": 1500.0})[0]["value"] == "1500"
    assert coerce_args(SPEC["set_pref"], {"name": "flag", "value": True})[0]["value"] == "true"


def test_coerce_drops_unknown_and_null_optional_args_with_warnings():
    args, warn = coerce_args(SPEC["reserve"], {"room_id": "R-9", "nights": None, "colour": "blue"})
    assert args == {"room_id": "R-9"}
    assert any("colour" in w for w in warn)


def test_uncoercible_value_is_left_for_the_kernel_to_reject():
    """Fail explicitly: we never guess a number out of prose; the kernel's schema gate rejects it."""
    args, warn = coerce_args(SPEC["find_rooms"], {"town": "A", "beds": "a couple", "budget": 10})
    assert args["beds"] == "a couple" and warn
    assert SPEC["find_rooms"].validate(args) == "'beds' must be a number"


# ── plan parsing ────────────────────────────────────────────────────────────
def test_parse_new_calls_and_reply():
    raw = json.dumps({"keep": [], "new_calls": [{"tool": "find_rooms", "args": {"town": "A", "beds": "1", "budget": 5}}],
                      "reply": "Let me look.", "reply_kind": "progress"})
    draft, warn = parse_plan(raw, snap(), SPEC)
    [c] = draft.calls
    assert (c.tool, c.args, draft.reply, draft.reply_kind) == ("find_rooms", {"town": "A", "beds": 1, "budget": 5.0},
                                                             "Let me look.", "progress")


def test_parse_keep_reuses_exact_existing_call_identity():
    calls = [{"key": "k0", "tool": "set_pref", "occurrence": 0, "args": {"name": "n", "value": "v"},
              "status": "succeeded", "result": {"ok": 1}, "error": None}]
    raw = json.dumps({"keep": [0], "new_calls": [{"tool": "reserve", "args": {"room_id": "R-1"}}], "reply": None})
    draft, _ = parse_plan(raw, snap(calls), SPEC)
    assert [(c.tool, c.args) for c in draft.calls] == [("set_pref", {"name": "n", "value": "v"}), ("reserve", {"room_id": "R-1"})]


def test_parse_drops_unknown_tools_and_bad_keep_indices():
    raw = json.dumps({"keep": [7, "x"], "new_calls": [{"tool": "launch_rocket", "args": {}}], "reply": "Hi",
                      "reply_kind": "shout"})
    draft, warn = parse_plan(raw, snap(), SPEC)
    assert draft.calls == () and draft.reply_kind == "final"
    assert any("launch_rocket" in w for w in warn) and any("keep" in w for w in warn)


def test_parse_accepts_code_fenced_json():
    draft, _ = parse_plan('```json\n{"keep": [], "new_calls": [], "reply": "Hello!"}\n```', snap(), SPEC)
    assert draft.reply == "Hello!"


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_parse_rejects_non_object_output(raw):
    with pytest.raises(PlanParseError):
        parse_plan(raw, snap(), SPEC)


def test_messages_carry_conversation_calls_and_schemas():
    s = snap([{"key": "k", "tool": "reserve", "occurrence": 0, "args": {"room_id": "R-2"}, "status": "failed",
               "result": None, "error": "taken"}],
             [{"role": "user", "text": "um reserve R-2"}, {"role": "agent", "text": "One moment.", "kind": "backchannel"}])
    msgs = build_messages(s, TOOLS_X)
    assert msgs[0]["role"] == "system" and '"find_rooms"' in msgs[0]["content"]
    payload = json.loads(msgs[1]["content"])
    assert payload["calls"][0] == {"index": 0, "tool": "reserve", "args": {"room_id": "R-2"}, "status": "failed",
                                   "result": None, "error": "taken"}
    assert payload["conversation"][1] == {"role": "assistant", "text": "One moment."}
