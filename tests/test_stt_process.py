"""Dedicated STT child process: the agent's event loop never runs (or waits on) Whisper decoding.

Regression for the validation finding: in-process Whisper decoding held the GIL and blocked the
LiveKit event loop for seconds. Here a *pure-Python* busy decode (worst case for the GIL) runs in
the child while the parent's loop keeps ticking.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time

import numpy as np
import pytest

from fdagent.voice.stt_process import SttProcessClient, SttProcessError, SttRequestError, SttTimeout

FACTORY = "tests.stt_fakes:ScriptedTranscriber"


def audio(code: float, n: int = 1600) -> np.ndarray:
    a = np.zeros(n, dtype=np.float32)
    a[0] = code
    return a


@pytest.fixture
def client():
    c = SttProcessClient("fake-model", threads=1, timeout_s=10.0, factory=FACTORY).start()
    yield c
    c.close()


def test_long_python_decoding_does_not_block_the_event_loop(client):
    async def main():
        worst = 0.0
        task = asyncio.create_task(client.atranscribe(audio(1.5)))  # 1.5 s GIL-holding decode (in the child)
        while not task.done():
            t = time.monotonic()
            await asyncio.sleep(0.01)
            worst = max(worst, time.monotonic() - t - 0.01)
        return task.result(), worst

    text, worst_lag = asyncio.run(main())
    assert text == "ok:1600"
    assert worst_lag < 0.15, f"event loop lagged {worst_lag:.3f}s while STT was decoding"


def test_multiple_segments_resolve_independently(client):
    async def main():
        return await asyncio.gather(client.atranscribe(audio(0.3, 1000)), client.atranscribe(audio(0.0, 2000)),
                                    client.atranscribe(audio(0.1, 3000)))

    assert asyncio.run(main()) == ["ok:1000", "ok:2000", "ok:3000"]  # each result matched to its own request


def test_request_failure_is_reported_and_the_process_keeps_serving(client):
    async def main():
        with pytest.raises(SttRequestError, match="decoder exploded"):
            await client.atranscribe(audio(-2))
        return await client.atranscribe(audio(0.0, 800))

    assert asyncio.run(main()) == "ok:800" and client.alive


def test_stuck_request_times_out_without_hanging_and_late_result_is_discarded(client):
    async def main():
        t = time.monotonic()
        with pytest.raises(SttTimeout):
            await client.atranscribe(audio(-3), timeout_s=0.3)
        elapsed = time.monotonic() - t
        nxt = await client.atranscribe(audio(0.0, 900))  # served after the slow one finishes
        return elapsed, nxt

    elapsed, nxt = asyncio.run(main())
    assert elapsed < 1.0 and nxt == "ok:900"


def test_crash_fails_pending_requests_fast_and_later_calls_fail_fast(client):
    async def main():
        t = time.monotonic()
        with pytest.raises(SttProcessError, match="exited unexpectedly"):
            await client.atranscribe(audio(-1))
        return time.monotonic() - t

    assert asyncio.run(main()) < 5.0
    assert not client.alive

    async def again():
        with pytest.raises(SttProcessError):
            await client.atranscribe(audio(0.0))

    asyncio.run(again())


def test_startup_failure_is_explicit_and_leaves_no_child():
    c = SttProcessClient("fake-model", factory="tests.stt_fakes:FailingLoad", start_timeout_s=60)
    with pytest.raises(SttProcessError, match="weights missing"):
        c.start()
    c.close()
    assert not [p for p in mp.active_children() if p.name == "fdagent-stt"]


def test_clean_shutdown_leaves_no_child_process():
    c = SttProcessClient("fake-model", threads=1, factory=FACTORY).start()
    pid = c.pid
    assert asyncio.run(c.atranscribe(audio(0.0, 100))) == "ok:100"
    c.close()
    c.close()  # idempotent
    assert not c.alive and c._proc.exitcode == 0
    assert pid not in [p.pid for p in mp.active_children()]

    async def after_close():
        with pytest.raises(SttProcessError):
            await c.atranscribe(audio(0.0))

    asyncio.run(after_close())
