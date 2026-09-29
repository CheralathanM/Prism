"""Fast understanding: deterministic, in-step, no models.

Deliberately small. Semantic understanding (which city, which date) belongs to the
reasoner. The fast path only decides things that must be decided *immediately* and
cheaply: is this a correction cue, is this segment filler-only.
"""

from __future__ import annotations

import re

_CORRECTION = re.compile(
    r"\b(actually|no,? wait|wait,? no|i mean|instead|scratch that|never ?mind|"
    r"sorry,? i meant|change (?:that|it) to|make (?:that|it)|rather|on second thought|"
    r"i changed my mind)\b",
    re.IGNORECASE,
)

_FILLERS = {
    "um", "umm", "uh", "uhh", "hmm", "hm", "er", "erm", "ah", "oh", "like", "so",
    "well", "okay", "ok", "mm", "mhm", "yeah", "right",
}


def correction_cue(text: str) -> str | None:
    m = _CORRECTION.search(text)
    return m.group(0).lower() if m else None


def is_filler_only(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return all(w in _FILLERS for w in words)
