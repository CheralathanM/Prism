"""Kernel configuration. Every value here is recorded in the journal header for reproducibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KernelConfig:
    # How long the user must stay silent after end-of-turn before intent counts as stable
    # and tools may be dispatched. Protects against "Rome — no wait, Milan" mid-turn pauses.
    stability_s: float = 0.6
    # If False, read-only tools may be dispatched speculatively before intent is stable.
    # Must stay True for FDB-v3: every executed call is scored, so a stale read is a fail.
    gate_read_only: bool = True
    # A speech segment that has ended (VAD) but whose transcript has not arrived blocks dispatch.
    # If no transcript arrives within this many seconds, the segment is closed explicitly.
    transcript_timeout_s: float = 10.0
    # Attempts per operation (first try included) for retriable failures.
    max_attempts: int = 2
    # Speak a short acknowledgement when a turn becomes stable (latency / no dead air).
    backchannel: bool = True
    backchannel_text: str = "Okay, one moment."
    fallback_text: str = "Sorry, I didn't catch that. Could you say it again?"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
