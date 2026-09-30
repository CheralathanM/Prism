"""Per-example validation summary (unofficial exact-match pass check + kernel/speech behaviour).
Times: seconds from harness stream start, monotonic-anchored. No credentials printed."""
import json
import sys
from collections import Counter
from pathlib import Path

V3 = Path("/mnt/c/Users/chera/Downloads/Samsung Prism/third_party/Full-Duplex-Bench/v3")
sys.path.insert(0, str(V3))
from evaluate_pass_rate import evaluate_scenario_pass  # noqa: E402

bench = {s["id"]: s for s in json.loads((V3 / "benchmark_data_v2.json").read_text())["scenarios"]}

for d in sys.argv[1:]:
    OUT = Path(d)
    print(f"\n################ {OUT.name}")
    rp = OUT / "result_fdagent.json"
    if not rp.exists():
        print("  NO RESULT FILE"); continue
    res = json.loads(rp.read_text())
    sc = bench[res["example_id"]]
    t0, window = res.get("stream_start_time"), res.get("latency", {}).get("input_duration_s")
    calls = res.get("actual_tool_calls") or []
    ev = evaluate_scenario_pass(sc, calls, use_llm=False)
    exp = [(c["function"], c["args"]) for c in sc["expected_tool_calls"]]
    act = [(c["function"], c["args"]) for c in calls]
    print(f"  status={res.get('status')} room={res.get('room_name')} window={window}s user_speech_end_rel={res.get('user_speech_end_rel')}")
    print(f"  expected: {exp}")
    print(f"  actual:   {act}")
    print(f"  exact-match (unofficial, no LLM judge): passed={ev['passed']} reason={ev['failure_reason']!r}")
    cnt = Counter(f for f, _ in act)
    print(f"  dispatch counts: {dict(cnt)} | expected counts: {dict(Counter(f for f, _ in exp))}")
    print(f"  perceived_total_latency={res.get('perceived_total_latency')} first_speech_s={res.get('latency', {}).get('first_speech_s')}")
    ch = res.get("asr_chunks") or []
    span = "none" if not ch else f"{ch[0]['timestamp'][0]}..{ch[-1]['timestamp'][1]} s"
    print(f"  captured agent speech in recording: {span}")
    print(f"  output transcript: {res.get('transcript')!r}")
    jp = OUT / "journal.jsonl"
    if not jp.exists():
        print("  NO JOURNAL"); continue
    recs = [json.loads(l) for l in jp.read_text().splitlines()]
    notes = [r for r in recs if r["kind"] == "note" and "t" in r and "wall" in r]
    anchor = (notes[0]["wall"] - notes[0]["t"]) if notes else 0
    mono = lambda t: round(t + anchor - t0, 2)  # noqa: E731
    steps = [r for r in recs if r["kind"] == "step"]
    seqs = [s["seq"] for s in steps]
    dec = Counter(x["kind"] for s in steps for x in s["decisions"])
    print(f"  kernel: steps={len(steps)} ordered={all(a < b for a, b in zip(seqs, seqs[1:]))} decisions={dict(dec)}")
    for s in steps:
        e = s["event"]
        for x in s["decisions"]:
            k = x["kind"]
            if k == "gate_blocked" and x.get("reason") == "transcript_pending":
                print(f"    {mono(s['ts']):>7}s gate_blocked transcript_pending {x.get('tool')}")
            if k in ("dispatch", "superseded", "barge_in", "proposal_rejected", "result_rejected", "reply_withheld",
                     "reply_held_until_stable", "invalid_call", "completion_blocked", "reasoner_failed", "revived",
                     "filler_ignored", "backchannel", "transcript_timeout", "transcript_deadline_extended", "transcript_failed"):
                info = {kk: x[kk] for kk in ("tool", "args", "reason", "call_id", "error", "text") if kk in x}
                print(f"    {mono(s['ts']):>7}s {k} {info}")
        if e["type"] == "UserTranscript":
            print(f"    {mono(s['ts']):>7}s transcript gen-bump: {e['text']!r}")
        if e["type"] == "ReasonerProposal":
            print(f"    {mono(s['ts']):>7}s plan {e['request_id']}: calls={[(c['tool'], c['args']) for c in e['calls']]} "
                  f"reply={e['reply']!r} kind={e['reply_kind']}")
    for n in notes:
        nm = n.get("name")
        if nm in ("speech_start", "speech_end", "speech_failed", "tts_first_audio", "tts_stream_fallback", "planner_attempt", "speech_dropped_stale"):
            info = {k: n[k] for k in ("speech_kind", "interrupted", "error", "mode", "request_id", "retry_count", "status",
                                      "attempt_s", "text") if k in n}
            after = " (AFTER WINDOW)" if window and mono(n["t"]) > window else ""
            print(f"    {mono(n['t']):>7}s {nm} {info}{after}")
    errs = [l for l in (OUT / "agent.log").read_text().splitlines() if '"level": "ERROR"' in l and "data channel" not in l]
    print(f"  agent ERROR lines (excluding teardown data-channel noise): {len(errs)}")
    for l in errs[:3]:
        print("   ", json.loads(l).get("message", "")[:200])
