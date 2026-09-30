"""Deterministic tool-argument normalization (applied to planner output before the kernel).

Only parameters that declare a semantic ``format`` in their ToolSpec are touched, and only
when the value parses unambiguously; anything else passes through unchanged, so the
kernel's schema gate still sees (and can reject) the original value.

Dates (``format="date"``): spoken calendar dates are canonicalized to "<Month> <day>"
(plus ", <year>" when a year was stated). Accepted variants: ordinal suffixes, "the"/"of",
abbreviated months, day-first order, a leading weekday. ISO dates (YYYY-MM-DD) are a
documented valid form and are preserved; relative dates ("tomorrow") are preserved.
"""

from __future__ import annotations

import re
from typing import Any

from fdagent.core.model import ToolSpec

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December")
_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # Feb 29 allowed without a year
_WEEKDAYS = {"monday", "mon", "tuesday", "tue", "tues", "wednesday", "wed", "thursday", "thu", "thur",
             "thurs", "friday", "fri", "saturday", "sat", "sunday", "sun"}
_FILLER = {"the", "of", "on"}
_DAY = re.compile(r"^(\d{1,2})(st|nd|rd|th)?$")
_YEAR = re.compile(r"^\d{4}$")


def normalize_date(value: str) -> str:
    """Canonical "<Month> <day>[, <year>]" for a parseable spoken date, else ``value`` unchanged."""
    if not isinstance(value, str):
        return value
    tokens = [t for t in re.split(r"[\s,.]+", value.strip().lower()) if t and t not in _FILLER]
    if tokens and tokens[0] in _WEEKDAYS:
        tokens = tokens[1:]
    year = None
    if tokens and _YEAR.match(tokens[-1]):
        year = tokens.pop()
    if len(tokens) != 2:
        return value
    a, b = tokens
    if a in _MONTHS and _DAY.match(b):
        month, day = _MONTHS[a], int(_DAY.match(b).group(1))
    elif b in _MONTHS and _DAY.match(a):
        month, day = _MONTHS[b], int(_DAY.match(a).group(1))
    else:
        return value
    if not 1 <= day <= _DAYS_IN_MONTH[month - 1]:
        return value
    out = f"{_MONTH_NAMES[month - 1]} {day}"
    return f"{out}, {year}" if year else out


_NORMALIZERS = {"date": normalize_date}


def normalize_args(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``args`` with format-aware normalization applied (pure, deterministic)."""
    out = dict(args)
    for p in spec.params:
        fn = _NORMALIZERS.get(p.format)
        if fn is not None and isinstance(out.get(p.name), str):
            out[p.name] = fn(out[p.name])
    return out
