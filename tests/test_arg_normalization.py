"""Deterministic tool-argument normalization (dates). Synthetic values only."""

from __future__ import annotations

import json

import pytest

from fdagent.adapters.arg_normalization import normalize_args, normalize_date
from fdagent.adapters.fdb_tools import FDB_TOOL_SPECS
from fdagent.core.model import ToolParam, ToolSpec, desired_call_key
from fdagent.providers.openai_reasoner import coerce_args, parse_plan


@pytest.mark.parametrize("spoken, canonical", [
    ("April 7th", "April 7"),
    ("april 7", "April 7"),
    ("7th April", "April 7"),
    ("7th of April", "April 7"),
    ("the 7th of April", "April 7"),
    ("April the 7th", "April 7"),
    ("Apr 7", "April 7"),
    ("Sept. 9th", "September 9"),
    ("9 September", "September 9"),
    ("Friday, April 7th", "April 7"),
    ("April 7th, 2031", "April 7, 2031"),
    ("7 April 2031", "April 7, 2031"),
    ("  November 22nd  ", "November 22"),
    ("Feb 29", "February 29"),
])
def test_spoken_dates_are_canonicalized(spoken, canonical):
    assert normalize_date(spoken) == canonical


@pytest.mark.parametrize("value", [
    "April 7",          # already canonical: preserved exactly
    "April 7, 2031",    # canonical with year: preserved
    "2031-04-07",       # ISO is a documented valid form: preserved
    "tomorrow",         # relative dates need a calendar context: preserved
    "next Friday",
    "April 31st",       # invalid day: left for the kernel/model, never guessed
    "February 30",
    "April",            # incomplete
    "3/3",              # ambiguous numeric form
    "",
    "Eastvale",         # not a date at all
])
def test_non_canonicalizable_values_are_preserved(value):
    assert normalize_date(value) == value


def test_normalization_is_idempotent():
    for v in ("April 7th", "7th of April", "April 7th, 2031"):
        once = normalize_date(v)
        assert normalize_date(once) == once


SPEC = ToolSpec("book_slot", "Book a slot", (ToolParam("day", "string", format="date"), ToolParam("label", "string")))


def test_only_date_formatted_params_are_touched():
    out = normalize_args(SPEC, {"day": "7th of April", "label": "7th of April"})
    assert out == {"day": "April 7", "label": "7th of April"}
    assert normalize_args(SPEC, {"label": "x"}) == {"label": "x"}


def test_planner_coercion_applies_normalization_and_keys_collapse():
    a, _ = coerce_args(SPEC, {"day": "April 7th", "label": "L"})
    b, _ = coerce_args(SPEC, {"day": "the 7th of April", "label": "L"})
    assert a == b == {"day": "April 7", "label": "L"}
    # Same intent spoken differently -> same desired-call key -> the kernel never runs it twice.
    assert desired_call_key("book_slot", a) == desired_call_key("book_slot", b)


def test_fdb_search_flights_date_is_normalized_through_parse_plan():
    specs = {s.name: s for s in FDB_TOOL_SPECS}
    assert [p.format for p in specs["search_flights"].params] == ["", "date"]
    raw = json.dumps({"keep": [], "new_calls": [{"tool": "search_flights",
                                                 "args": {"destination": "Northfield", "date": "Oct 21st"}}]})
    snap = {"conversation": [], "calls": [], "orphan_effects": []}
    draft, _ = parse_plan(raw, snap, specs)
    assert draft.calls[0].args == {"destination": "Northfield", "date": "October 21"}
