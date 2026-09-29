"""Deterministic id generation (per session counters) so journals replay byte-for-byte."""

from __future__ import annotations

from collections import Counter


class IdGen:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._n: Counter[str] = Counter()

    def next(self, prefix: str) -> str:
        self._n[prefix] += 1
        return f"{prefix}-{self._n[prefix]}"
