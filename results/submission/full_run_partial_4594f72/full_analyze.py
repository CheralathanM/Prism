"""Analyse the full FDB-v3 run (evaluation side only; reads benchmark expectations to score).

Outputs summary.json and summary.md next to this file. Scoring = official evaluate_scenario_pass
with use_llm=False (exact match; unofficial). Categories per failing recording:
  harness/infrastructure, planner, tool selection/arguments, STT/transcription, timing/latency
plus secondary flags (TTS/audio, gemini errors, ...). No credentials are read or printed.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
V3 = REPO / "third_party" / "Full-Duplex-Bench" / "v3"
sys.path.insert(0, str(V3))
from evaluate_pass_rate import evaluate_scenario_pass  # noqa: E402

BENCH = {s["id"]: s for s in json.loads((V3 / "benchmark_data_v2.json").read_text(encoding="utf-8"))["scenarios"]}
PROGRESS_SECS = {}
if (RUN / "progress.log").exists():
    for _line in (RUN / "progress.log").read_text().splitlines():
        _m = re.search(r" (\S+) harness_rc=.* secs=(\d+)", _line)
        if _m:
            PROGRESS_SECS[_m.group(1)] = int(_m.group(2))

NUMWORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "hundred": 100, "thousand": 1000}


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def value_heard(value, text: str) -> bool:
    """Is an expected argument value recognisably present in a transcript?"""
    t = norm(text)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value) if float(value).is_integer() else value
        digits = re.sub(r"[^0-9]", "", str(n))
        return bool(digits) and digits in re.sub(r"[^0-9]", "", text.replace(",", ""))
    v = norm(value)
    if not v:
        return True
    if v in t or v.replace(" ", "") in t.replace(" ", ""):
        return True
    # date-like "july 15" vs "july 15th"
    m = re.match(r"([a-z]+) (\d+)$", v)
    return bool(m and re.search(rf"{m.group(1)} {m.group(2)}(st|nd|rd|th)?\b", t))


def gemini_class(err: str) -> str | None:
    e = err.lower()
    if "429" in e or "rate" in e and "limit" in e or "quota" in e or "resource_exhausted" in e:
        return "gemini_rate_limit"
    if any(k in e for k in ("503", "500", "502", "504", "unavailable", "internalservererror", "apiconnection",
                            "timeout", "connecterror")):
        return "gemini_service"
    return None


def analyse(d: Path) -> dict:
    r: dict = {"folder": d.name, "example_id": d.name.rsplit("_", 1)[0]}
    sc = BENCH.get(r["example_id"])
    r["difficulty"], r["domain"] = (sc or {}).get("difficulty"), (sc or {}).get("domain")
    r["features"] = (sc or {}).get("disfluency_features", [])
    r["rollback"] = bool((sc or {}).get("state_rollback_test"))
    flags: list[str] = []
    harness_log = (d / "harness.log").read_text(errors="replace") if (d / "harness.log").exists() else ""
    m = re.search(r"livekit_inference\.py failed with exit code (-?\d+)", harness_log)
    if m:
        flags.append(f"harness_client_exit_{m.group(1)}")
    if (d / "infra_error").exists():
        flags.append("agent_not_ready")
    agent_log = (d / "agent.log").read_text(errors="replace") if (d / "agent.log").exists() else ""
    if "assignment for job" in agent_log and "timed out" in agent_log or "process is unresponsive" in agent_log:
        flags.append("job_assignment_timeout_or_host_suspend")
    secs = PROGRESS_SECS.get(d.name)
    r["wall_secs"] = secs
    if secs and secs > 600:
        flags.append("abnormally_long_recording(host_suspend?)")
    res = json.loads((d / "result_fdagent.json").read_text()) if (d / "result_fdagent.json").exists() else None
    r["status"] = res.get("status") if res else "no_result"

    # ── journal (recovered from the room named in harness.log if the harness crashed) ──
    jp = d / "journal.jsonl"
    room = (res or {}).get("room_name") or (re.search(r"into room: (eval-[0-9a-f]+)", harness_log) or [None, None])[1]
    r["room"] = room
    if not jp.exists() and room and (REPO / "results" / "journals" / f"{room}.jsonl").exists():
        jp = REPO / "results" / "journals" / f"{room}.jsonl"
        r["journal_recovered"] = True
    recs = [json.loads(l) for l in jp.read_text().splitlines()] if jp.exists() else []
    r["agent_joined"] = bool(recs)
    steps = [x for x in recs if x["kind"] == "step"]
    notes = [x for x in recs if x["kind"] == "note"]
    decisions = [(s["ts"], dd) for s in steps for dd in s["decisions"]]
    transcripts = [s["event"]["text"] for s in steps if s["event"]["type"] == "UserTranscript" and s["event"].get("final")]
    our_text = " ".join(transcripts)
    r["our_transcript"] = our_text
    dispatches = [dd for _, dd in decisions if dd["kind"] == "dispatch"]
    r["dispatches"] = [(dd["tool"], dd["args"]) for dd in dispatches]
    r["barge_in"] = sum(dd["kind"] == "barge_in" for _, dd in decisions)
    r["superseded_ops"] = sum(dd["kind"] == "superseded" for _, dd in decisions)
    r["transcript_timeouts"] = sum(dd["kind"] == "transcript_timeout" for _, dd in decisions)
    r["transcript_failed"] = sum(dd["kind"] == "transcript_failed" for _, dd in decisions)
    r["deadline_extended"] = sum(dd["kind"] == "transcript_deadline_extended" for _, dd in decisions)
    reasoner_errors = [dd.get("error", "") for _, dd in decisions if dd["kind"] == "reasoner_failed"]
    r["reasoner_errors"] = reasoner_errors
    attempts = [n for n in notes if n.get("name") == "planner_attempt"]
    r["planner_attempts"] = len(attempts)
    r["planner_status"] = dict(Counter(str(a.get("status")) for a in attempts))
    r["planner_retries"] = sum(1 for a in attempts if a.get("retry_count") not in (None, "0"))
    r["planner_429"] = sum(1 for a in attempts if a.get("status") == 429)
    r["planner_max_attempt_s"] = max([a.get("attempt_s") or 0 for a in attempts], default=None)
    r["speech_failed"] = sum(1 for n in notes if n.get("name") == "speech_failed")
    r["speech_dropped_stale"] = sum(1 for n in notes if n.get("name") == "speech_dropped_stale")
    r["tts_fallbacks"] = sum(1 for n in notes if n.get("name") == "tts_stream_fallback")
    r["interrupted_speech"] = sum(1 for n in notes if n.get("name") == "speech_end" and n.get("interrupted"))
    for e in reasoner_errors:
        g = gemini_class(e)
        if g:
            flags.append(g)
    if r["planner_429"]:
        flags.append("gemini_rate_limit")

    # ── harness result ──
    calls = (res or {}).get("actual_tool_calls") or []
    r["tool_calls"] = [(c["function"], c["args"]) for c in calls]
    r["perceived_latency_s"] = (res or {}).get("perceived_total_latency")
    r["first_speech_s"] = ((res or {}).get("latency") or {}).get("first_speech_s")
    r["window_s"] = ((res or {}).get("latency") or {}).get("input_duration_s")
    chunks = (res or {}).get("asr_chunks") or []
    r["speech_captured"] = bool(chunks)
    r["agent_output_transcript"] = (res or {}).get("transcript", "")
    if not r["speech_captured"] and r["status"] == "completed":
        flags.append("no_speech_captured")
    if r["speech_failed"]:
        flags.append("tts_speech_failed")

    # duplicates / premature
    cnt = Counter(json.dumps([f, a], sort_keys=True) for f, a in r["tool_calls"])
    r["identical_duplicates"] = sum(v - 1 for v in cnt.values() if v > 1)
    exp = Counter(c["function"] for c in (sc or {}).get("expected_tool_calls", []))
    act = Counter(f for f, _ in r["tool_calls"])
    r["extra_calls"] = sum(max(0, act[f] - exp.get(f, 0)) for f in act)
    by_fn: dict = {}
    for f, a in r["tool_calls"]:
        by_fn.setdefault(f, []).append(a)
    r["premature_dispatch"] = sum(1 for f, lst in by_fn.items() if len(lst) > exp.get(f, 0) and len(set(json.dumps(x, sort_keys=True) for x in lst)) > 1)

    # supersession (rollback scenarios)
    if r["rollback"] and sc:
        det = sc.get("state_rollback_details") or {}
        orig, corr = det.get("original_param", {}), det.get("corrected_param", {})
        dispatched_text = json.dumps(r["tool_calls"]).lower()
        stale = [k for k, v in orig.items() if corr.get(k) != v and norm(v) and norm(v) in norm(dispatched_text)]
        r["rollback_outcome"] = "stale_value_executed" if stale else ("corrected_only" if r["tool_calls"] else "no_call")

    # ── scoring + category ──
    if r["status"] != "completed" or "agent_not_ready" in flags:
        r["passed"], r["category"], r["reason"] = None, "harness/infrastructure", r["status"]
    elif not r["agent_joined"] and "job_assignment_timeout_or_host_suspend" in flags:
        # The harness finished, but our agent was never assigned the job (host suspended mid-assignment).
        r["passed"], r["category"], r["reason"] = None, "harness/infrastructure", "agent never joined (job assignment timeout / host suspend)"
    else:
        ev = evaluate_scenario_pass(sc, calls, use_llm=False)
        r["passed"], r["reason"] = ev["passed"], ev["failure_reason"]
        if ev["passed"]:
            r["category"] = "pass"
        else:
            expected_vals = [v for c in sc["expected_tool_calls"] for v in c["args"].values()
                             if not (isinstance(v, str) and v.startswith("$"))]
            spoken = " ".join(t["user"] for t in sc["dialogue"])
            missing_in_stt = [v for v in expected_vals if value_heard(v, spoken) and not value_heard(v, our_text)]
            r["values_missing_in_our_transcript"] = missing_in_stt
            window = r["window_s"] or 0
            if not calls and any(gemini_class(e) for e in reasoner_errors):
                r["category"] = "harness/infrastructure"
            elif missing_in_stt:
                r["category"] = "STT/transcription"
            elif not calls and not dispatches:
                late = (steps and window and transcripts and
                        (steps[-1]["ts"] - steps[0]["ts"]) > window)
                r["category"] = "timing/latency" if late and not our_text else "planner"
            elif not calls and dispatches:
                r["category"] = "timing/latency"  # dispatched, but not logged within the harness session
            else:
                r["category"] = "tool selection/arguments"
    r["flags"] = sorted(set(flags))
    return r


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def main():
    # Only completed recordings (a "done" marker); an interrupted recording is excluded (see PARTIAL_RUN.md).
    rows = [analyse(d) for d in sorted((RUN / "examples").iterdir()) if d.is_dir() and (d / "done").exists()]
    completed = [r for r in rows if r["status"] == "completed" and r["category"] != "harness/infrastructure"]
    scored = [r for r in rows if r["passed"] is not None]
    lat = [r["perceived_latency_s"] for r in rows if isinstance(r["perceived_latency_s"], (int, float))]
    summary = {
        "recordings": len(rows),
        "completed": sum(r["status"] == "completed" for r in rows),
        "scored": len(scored),
        "exact_match_pass": sum(bool(r["passed"]) for r in scored),
        "exact_match_fail": sum(r["passed"] is False for r in scored),
        "categories": dict(Counter(r["category"] for r in rows)),
        "identical_duplicate_calls": sum(r["identical_duplicates"] for r in rows),
        "extra_calls": sum(r["extra_calls"] for r in rows),
        "recordings_with_premature_dispatch": sum(1 for r in rows if r["premature_dispatch"]),
        "rollback_outcomes": dict(Counter(r.get("rollback_outcome") for r in rows if r["rollback"])),
        "barge_in_recordings": sum(1 for r in rows if r["barge_in"]),
        "barge_in_events": sum(r["barge_in"] for r in rows),
        "stale_speech_dropped": sum(r["speech_dropped_stale"] for r in rows),
        "speech_captured": sum(r["speech_captured"] for r in rows),
        "tts_speech_failed_events": sum(r["speech_failed"] for r in rows),
        "tts_stream_fallbacks": sum(r["tts_fallbacks"] for r in rows),
        "perceived_latency_s": {"n": len(lat), "median": statistics.median(lat) if lat else None,
                                "mean": round(statistics.mean(lat), 2) if lat else None,
                                "p90": pct(lat, 0.9), "min": min(lat, default=None), "max": max(lat, default=None)},
        "planner": {"attempts": sum(r["planner_attempts"] for r in rows),
                    "status_counts": dict(sum((Counter(r["planner_status"]) for r in rows), Counter())),
                    "retried_attempts": sum(r["planner_retries"] for r in rows),
                    "http_429": sum(r["planner_429"] for r in rows),
                    "reasoner_failures": sum(len(r["reasoner_errors"]) for r in rows),
                    "slowest_attempt_s": max([r["planner_max_attempt_s"] or 0 for r in rows], default=None)},
        "transcription": {"timeouts": sum(r["transcript_timeouts"] for r in rows),
                          "failures": sum(r["transcript_failed"] for r in rows),
                          "deadline_extensions": sum(r["deadline_extended"] for r in rows)},
        "harness_issues": [(r["folder"], r["status"], r["flags"]) for r in rows
                           if r["category"] == "harness/infrastructure" or any(f.startswith("harness") for f in r["flags"])],
        "by_difficulty": {k: f"{sum(bool(r['passed']) for r in scored if r['difficulty'] == k)}/"
                             f"{sum(1 for r in scored if r['difficulty'] == k)}" for k in ("easy", "medium", "hard")},
        "by_feature": {f: f"{sum(bool(r['passed']) for r in scored if f in r['features'])}/"
                          f"{sum(1 for r in scored if f in r['features'])}"
                       for f in sorted({f for r in scored for f in r['features']})},
    }
    (RUN / "summary.json").write_text(json.dumps({"summary": summary, "recordings": rows}, indent=2, default=str))
    lines = ["# Full FDB-v3 run summary (commit 4594f72, unofficial exact-match scoring)", "",
             "```", json.dumps(summary, indent=2, default=str), "```", "",
             "| recording | category | passed | tool calls | perceived latency | speech | flags | reason |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['folder']} | {r['category']} | {r['passed']} | {len(r['tool_calls'])} | "
                     f"{r['perceived_latency_s']} | {'yes' if r['speech_captured'] else 'no'} | "
                     f"{', '.join(r['flags'])} | {str(r.get('reason', ''))[:80]} |")
    (RUN / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
