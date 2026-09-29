"""Benchmark integrity: agent code must not contain FDB-v3 test items or expected answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdagent.adapters.fdb_tools import DEFAULT_FDB_V3_DIR

PKG = Path(__file__).resolve().parents[1] / "fdagent"
BENCH = DEFAULT_FDB_V3_DIR / "benchmark_data_v2.json"


@pytest.mark.skipif(not BENCH.exists(), reason="FDB-v3 not fetched")
def test_no_scenario_ids_or_expected_arguments_in_agent_code():
    scenarios = json.loads(BENCH.read_text(encoding="utf-8"))["scenarios"]
    source = "\n".join(p.read_text(encoding="utf-8") for p in PKG.rglob("*.py")).lower()
    hits = set()
    for s in scenarios:
        if s["id"].lower() in source:
            hits.add(s["id"])
        for call in s["expected_tool_calls"]:
            for v in call["args"].values():
                # Long, specific string values only (short generic words like "gold" are tool vocabulary).
                if isinstance(v, str) and len(v) >= 12 and not v.startswith("$") and v.lower() in source:
                    hits.add(f"{s['id']}:{v}")
        for turn in s["dialogue"]:
            if turn["user"][:60].lower() in source:
                hits.add(f"{s['id']}:utterance")
    assert not hits, f"benchmark content found in agent code: {sorted(hits)[:10]}"
