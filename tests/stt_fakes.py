"""Transcriber fakes loaded *inside* the spawned STT child process (must stay import-light).

ScriptedTranscriber: the first audio sample selects the behaviour of a request:
  v >= 0  -> busy pure-Python loop for v seconds (holds the child's GIL, like a long decode),
             then returns "ok:<n samples>"
  -1      -> os._exit(3) (hard crash)
  -2      -> raise ValueError (per-request failure)
  -3      -> sleep 1.0 s, then return "slow" (slow/stuck request)
"""

from __future__ import annotations

import os
import time


class ScriptedTranscriber:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def load(self) -> None:
        pass

    def transcribe(self, audio) -> str:
        v = float(audio[0]) if len(audio) else 0.0
        if v == -1:
            os._exit(3)
        if v == -2:
            raise ValueError("decoder exploded")
        if v == -3:
            time.sleep(1.0)
            return "slow"
        end = time.monotonic() + v
        x = 0
        while time.monotonic() < end:  # pure Python: holds this process's GIL
            x += 1
        return f"ok:{len(audio)}"


class FailingLoad:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def load(self) -> None:
        raise RuntimeError("weights missing")
