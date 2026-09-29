"""Deterministic offline replay: feed a journal's events into a fresh kernel and diff actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import KernelConfig
from .events import event_from_dict
from .inbox import Envelope
from .journal import Journal, tool_spec_from_dict
from .kernel import SessionKernel


def _plain(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def replay(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a list of mismatches (empty list = byte-identical decisions and actions)."""
    header = records[0]
    assert header["kind"] == "header", "journal must start with a header"
    kernel = SessionKernel(
        header["session_id"],
        [tool_spec_from_dict(t) for t in header["tools"]],
        KernelConfig(**header["config"]),
        journal=Journal(),
    )
    mismatches = []
    for rec in records[1:]:
        if rec["kind"] != "step":
            continue
        env = Envelope(rec["seq"], rec["ts"], event_from_dict(rec["event"]))
        actions = _plain([a.to_dict() for a in kernel.step(env)])
        decisions = kernel.journal.records[-1]["decisions"]
        if actions != rec["actions"] or decisions != rec["decisions"]:
            mismatches.append({"seq": rec["seq"], "expected": rec["actions"], "got": actions})
    return mismatches


def main() -> None:
    import sys

    path = Path(sys.argv[1])
    mm = replay(Journal.load(path))
    print(f"{path}: {'OK — replay identical' if not mm else f'{len(mm)} mismatching steps'}")
    for m in mm[:5]:
        print(json.dumps(m, indent=1)[:2000])
    sys.exit(1 if mm else 0)


if __name__ == "__main__":
    main()
