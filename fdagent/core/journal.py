"""Append-only journal: header + one record per kernel step (event, decisions, actions).

Enough to answer "why did this tool call execute / why was this result rejected / why
was this superseded / why did the system say this", and to replay a session offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any, Iterable

from .actions import Action
from .config import KernelConfig
from .inbox import Envelope
from .model import ToolParam, ToolSpec


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


class Journal:
    def __init__(self, path: str | Path | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self._fh: IO[str] | None = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(p, "w", encoding="utf-8")

    def _write(self, rec: dict[str, Any]) -> None:
        rec = _plain(rec)
        self.records.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def header(self, session_id: str, config: KernelConfig, tools: Iterable[ToolSpec], **meta: Any) -> None:
        self._write(
            {
                "kind": "header",
                "session_id": session_id,
                "config": config.to_dict(),
                "tools": [asdict(t) for t in tools],
                **meta,
            }
        )

    def record(self, env: Envelope, decisions: list[dict[str, Any]], actions: list[Action]) -> None:
        self._write(
            {
                "kind": "step",
                "seq": env.seq,
                "ts": env.ts,
                "event": env.event.to_dict(),
                "decisions": decisions,
                "actions": [a.to_dict() for a in actions],
            }
        )

    def annotate(self, **fields: Any) -> None:
        """Out-of-kernel observations (latency marks, executor notes). Not replayed."""
        if "kind" in fields:
            raise ValueError("'kind' is reserved in journal records")
        self._write({"kind": "note", **fields})

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def tool_spec_from_dict(d: dict[str, Any]) -> ToolSpec:
    d = dict(d)
    d["params"] = tuple(ToolParam(**p) for p in d["params"])
    return ToolSpec(**d)
