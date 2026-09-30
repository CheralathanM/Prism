"""Benchmark integrity: agent code must not contain FDB-v3 test items or expected answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdagent.adapters.fdb_tools import DEFAULT_FDB_V3_DIR

PKG = Path(__file__).resolve().parents[1] / "fdagent"
BENCH = DEFAULT_FDB_V3_DIR / "benchmark_data_v2.json"


def test_no_utf8_bom_in_repo_text_files():
    """Windows tooling can prepend a BOM; TOML and shell scripts on Linux reject it."""
    root = PKG.parent
    bad = [
        str(p.relative_to(root))
        for pattern in ("*.py", "*.toml", "*.sh", "*.md", "*.txt", ".gitignore", ".gitattributes", ".env.example")
        for p in root.rglob(pattern)
        if not {".venv", ".git", "third_party"} & set(p.relative_to(root).parts)
        and p.read_bytes().startswith(b"\xef\xbb\xbf")
    ]
    assert not bad, bad


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


@pytest.mark.skipif(not BENCH.exists(), reason="FDB-v3 not fetched")
def test_no_benchmark_argument_values_in_tests():
    """Tests must use synthetic values too: no expected or rolled-back argument values
    (whole-word, >= 3 chars). Tool parameter names are schema vocabulary, not answers."""
    import re

    from fdagent.adapters.fdb_tools import FDB_TOOL_SPECS

    param_names = {p.name.lower() for s in FDB_TOOL_SPECS for p in s.params}
    scenarios = json.loads(BENCH.read_text(encoding="utf-8"))["scenarios"]
    values = set()
    for s in scenarios:
        for call in s["expected_tool_calls"]:
            values |= {v for v in call["args"].values() if isinstance(v, str)}
        for side in ("original_param", "corrected_param"):
            values |= {v for v in (s.get("state_rollback_details") or {}).get(side, {}).values() if isinstance(v, str)}
    values = {v for v in values if len(v) >= 3 and not v.startswith("$") and v.lower() not in param_names}
    values |= {s["id"] for s in scenarios}  # scenario identifiers don't belong in tests either
    hits = {}
    for p in Path(__file__).parent.glob("*.py"):
        if p.name == Path(__file__).name:
            continue
        src = p.read_text(encoding="utf-8")
        found = sorted(v for v in values if re.search(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z0-9])", src, re.I))
        if found:
            hits[p.name] = found
    assert not hits, hits
