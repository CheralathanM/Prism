"""Timers: sleep, then post TimerFired. The kernel decides whether the timer is still relevant."""

from __future__ import annotations

import asyncio
from typing import Callable

from fdagent.core.actions import StartTimer
from fdagent.core.events import Event, TimerFired


class Timers:
    def __init__(self, post: Callable[[Event], object]) -> None:
        self._post = post
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, a: StartTimer) -> None:
        old = self._tasks.pop(a.timer_id, None)
        if old:
            old.cancel()
        self._tasks[a.timer_id] = asyncio.create_task(self._fire(a), name=f"timer:{a.timer_id}")

    async def _fire(self, a: StartTimer) -> None:
        await asyncio.sleep(a.delay_s)
        self._tasks.pop(a.timer_id, None)
        self._post(TimerFired(a.timer_id, a.kind, a.generation, a.call_id, a.attempt, a.epoch))

    async def aclose(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
