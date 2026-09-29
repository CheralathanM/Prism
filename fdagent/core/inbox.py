"""Ordered inbox: the single sequence of truth for everything the kernel sees."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from .events import Event


@dataclass(frozen=True)
class Envelope:
    seq: int  # monotonically increasing; defines processing order
    ts: float  # clock reading at enqueue (the kernel never reads a clock itself)
    event: Event


class Inbox:
    """FIFO with sequence numbers assigned at enqueue time.

    All producers (ingress callbacks, workers, timers) run on the same asyncio loop and
    call ``put`` synchronously, so the enqueue order is well defined. Producers on other
    threads must hop onto the loop with ``loop.call_soon_threadsafe(inbox.put, ev)``.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seq = 0
        self._q: asyncio.Queue[Envelope] = asyncio.Queue()

    def put(self, event: Event) -> Envelope:
        self._seq += 1
        env = Envelope(self._seq, self._clock(), event)
        self._q.put_nowait(env)
        return env

    async def get(self) -> Envelope:
        return await self._q.get()

    def get_nowait(self) -> Envelope | None:
        try:
            return self._q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def __len__(self) -> int:
        return self._q.qsize()
