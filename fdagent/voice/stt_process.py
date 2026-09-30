"""Dedicated STT child process: Whisper never runs inside the LiveKit agent process.

Why: transformers' Whisper decoding is largely Python code; run in a thread of the agent
process it held the GIL and starved the LiveKit event loop (measured up to ~6.8 s of loop
blocking per example, 30 s transcription backlogs, late timers). A separate process has its own
interpreter and GIL, so the agent loop only does cheap IPC.

Parent side (``SttProcessClient``):
  * ``start()`` spawns the child and blocks until the model is loaded (called in prewarm).
  * ``await atranscribe(audio)`` sends float32 16 kHz mono audio and awaits the text. Requests
    are independent (ids); several may be in flight. Sends go through a multiprocessing.Queue
    (its feeder thread writes the pipe), results come back via a reader thread that resolves
    asyncio futures with ``call_soon_threadsafe``; the event loop never blocks on IPC.
  * Failures never hang the agent: a per-request timeout raises ``SttTimeout``; a crashed
    child fails every pending request with ``SttProcessError`` and later calls fail fast.
  * ``close()`` asks the child to exit, then terminates/kills it if needed. Also registered
    with ``atexit``; the child is a daemon and exits if its parent disappears.

Child side (``worker_main``): caps torch threads, loads the transcriber, reports ``ready``,
then serves requests one at a time.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import itertools
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any

import numpy as np

log = logging.getLogger("fdagent.stt_process")

DEFAULT_FACTORY = "fdagent.voice.whisper_core:WhisperTranscriber"


class SttProcessError(RuntimeError):
    """The STT process is unavailable (failed to start, crashed, or closed)."""


class SttTimeout(SttProcessError):
    """A single request exceeded its timeout (the process may be stuck)."""


class SttRequestError(RuntimeError):
    """The STT process ran but transcription of this segment failed."""


def _resolve(spec: str) -> Any:
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def worker_main(req_q: Any, res_q: Any, model_id: str, threads: int, factory: str, parent_pid: int) -> None:
    """Entry point of the STT child process."""
    try:
        if threads > 0:
            try:
                import torch
                torch.set_num_threads(threads)
            except ImportError:
                pass
        t0 = time.monotonic()
        transcriber = _resolve(factory)(model_id)
        transcriber.load()
        res_q.put(("ready", round(time.monotonic() - t0, 2)))
    except BaseException as e:  # noqa: BLE001 - report every startup failure to the parent
        res_q.put(("fatal", f"{type(e).__name__}: {e}"))
        return
    while True:
        try:
            msg = req_q.get(timeout=1.0)
        except queue.Empty:
            if os.getppid() != parent_pid:  # parent is gone: never linger as an orphan
                return
            continue
        if msg[0] == "close":
            return
        _, rid, raw = msg
        t = time.monotonic()
        try:
            text = transcriber.transcribe(np.frombuffer(raw, dtype=np.float32))
            res_q.put(("ok", rid, text, round(time.monotonic() - t, 3)))
        except Exception as e:  # noqa: BLE001 - per-request failure; keep serving
            res_q.put(("err", rid, f"{type(e).__name__}: {e}", round(time.monotonic() - t, 3)))


class SttProcessClient:
    def __init__(self, model_id: str, threads: int = 4, timeout_s: float = 60.0, start_timeout_s: float = 600.0,
                 factory: str = DEFAULT_FACTORY) -> None:
        self.model_id = model_id
        self.threads = threads
        self.timeout_s = timeout_s
        self.start_timeout_s = start_timeout_s
        self.factory = factory
        self._ctx = mp.get_context("spawn")  # never fork a process that has threads
        self._proc: Any = None
        self._req_q: Any = None
        self._res_q: Any = None
        self._pending: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._reader: threading.Thread | None = None
        self._closed = threading.Event()
        self._dead: str | None = None
        self.load_s: float | None = None
        self.last_duration_s: float | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive() and self._dead is None and not self._closed.is_set()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def load(self) -> "SttProcessClient":  # prewarm-compatible alias
        return self.start()

    def start(self) -> "SttProcessClient":
        if self._proc is not None:
            return self
        self._req_q, self._res_q = self._ctx.Queue(), self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=worker_main, name="fdagent-stt", daemon=True,
            args=(self._req_q, self._res_q, self.model_id, self.threads, self.factory, os.getpid()))
        self._proc.start()
        deadline = time.monotonic() + self.start_timeout_s
        while True:
            try:
                msg = self._res_q.get(timeout=0.5)
            except queue.Empty:
                if not self._proc.is_alive():
                    self._dead = f"STT process exited during startup (code {self._proc.exitcode})"
                    raise SttProcessError(self._dead) from None
                if time.monotonic() > deadline:
                    self.close()
                    raise SttProcessError(f"STT process did not load {self.model_id} within {self.start_timeout_s}s") from None
                continue
            if msg[0] == "ready":
                self.load_s = msg[1]
                break
            if msg[0] == "fatal":
                self._dead = f"STT process failed to start: {msg[1]}"
                self._proc.join(5)
                raise SttProcessError(self._dead)
        self._reader = threading.Thread(target=self._read_results, name="fdagent-stt-reader", daemon=True)
        self._reader.start()
        atexit.register(self.close)
        log.info("STT process ready pid=%s model=%s load_s=%s", self._proc.pid, self.model_id, self.load_s)
        return self

    def close(self, timeout_s: float = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        proc = self._proc
        if proc is not None and proc.is_alive():
            try:
                self._req_q.put(("close",))
            except (OSError, ValueError):
                pass
            proc.join(timeout_s)
            if proc.is_alive():
                proc.terminate()
                proc.join(2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(2.0)
        self._fail_all(SttProcessError("STT process closed"))
        for q in (self._req_q, self._res_q):
            if q is not None:
                q.close()
                q.cancel_join_thread()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(2.0)

    # ── requests ───────────────────────────────────────────────────────────
    async def atranscribe(self, audio: np.ndarray, timeout_s: float | None = None) -> str:
        if not self.alive:
            raise SttProcessError(self._dead or "STT process is not running")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        rid = next(self._ids)
        with self._lock:
            self._pending[rid] = (loop, fut)
        # Queue.put is non-blocking here: a feeder thread pickles and writes to the pipe.
        self._req_q.put(("transcribe", rid, np.ascontiguousarray(audio, dtype=np.float32).tobytes()))
        try:
            return await asyncio.wait_for(fut, timeout_s if timeout_s is not None else self.timeout_s)
        except asyncio.TimeoutError:
            raise SttTimeout(f"STT request {rid} exceeded {timeout_s or self.timeout_s}s") from None
        finally:
            with self._lock:
                self._pending.pop(rid, None)

    def _read_results(self) -> None:
        while not self._closed.is_set():
            try:
                msg = self._res_q.get(timeout=0.5)
            except queue.Empty:
                if self._proc is not None and not self._proc.is_alive():
                    self._dead = f"STT process exited unexpectedly (code {self._proc.exitcode})"
                    log.error(self._dead)
                    self._fail_all(SttProcessError(self._dead))
                    return
                continue
            except (EOFError, OSError, ValueError):
                return
            kind, rid = msg[0], msg[1]
            with self._lock:
                entry = self._pending.get(rid)
            if entry is None:
                continue  # timed out earlier; late result discarded
            loop, fut = entry
            self.last_duration_s = msg[3] if len(msg) > 3 else None
            if kind == "ok":
                loop.call_soon_threadsafe(_set_result, fut, msg[2])
            else:
                loop.call_soon_threadsafe(_set_exception, fut, SttRequestError(msg[2]))

    def _fail_all(self, exc: Exception) -> None:
        with self._lock:
            entries = list(self._pending.values())
        for loop, fut in entries:
            try:
                loop.call_soon_threadsafe(_set_exception, fut, exc)
            except RuntimeError:
                pass  # that event loop is already closed


def _set_result(fut: asyncio.Future, value: Any) -> None:
    if not fut.done():
        fut.set_result(value)


def _set_exception(fut: asyncio.Future, exc: Exception) -> None:
    if not fut.done():
        fut.set_exception(exc)
