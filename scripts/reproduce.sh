#!/usr/bin/env bash
# One-command FDB-v3 reproduction for the fdagent LiveKit agent.
#
#   bash scripts/reproduce.sh                        # all 100 recordings
#   bash scripts/reproduce.sh --example travel_01    # one scenario (all its recordings)
#
# Steps: check prerequisites -> fetch FDB-v3 (pinned) -> for each recording start a FRESH agent,
# run the official, unmodified run_tool_benchmark.py (--provider fdagent), stop the agent, archive all
# outputs under results/runs/<run_id>/ (generated files are moved out of the input folders) ->
# official evaluation: with OPENAI_API_KEY -> --use-llm (official gpt-4o judge, paid); without ->
# exact-match fallback (UNOFFICIAL).
#
# Requirements: Linux, python venv with requirements*.txt installed (see README), ffmpeg,
# .env.local with LIVEKIT_* and GOOGLE_API_KEY, benchmark audio in the FDB-v3 v3/ folder.
# CPU-only machines: FDAGENT_CPU_HARNESS_SHIM=1 (see README).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
V3="$REPO/third_party/Full-Duplex-Bench/v3"
DATA="$V3/fdb_v3_data_released"
PY="${PYTHON:-python}"
ONLY=""
[ "${1:-}" = "--example" ] && ONLY="${2:?--example needs an id}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$REPO/results/runs/$RUN_ID"
mkdir -p "$RUN/examples"

[ -f "$REPO/.env.local" ] || { echo "missing .env.local (copy .env.example)"; exit 2; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 2; }
bash "$REPO/scripts/fetch_fdb.sh" || exit 2
[ -d "$DATA" ] || { echo "benchmark audio missing at $DATA (see v3/README.md)"; exit 2; }

set -a; . "$REPO/.env.local"; set +a
export PYTHONUNBUFFERED=1
HARNESS_PYTHONPATH=""
[ "${FDAGENT_CPU_HARNESS_SHIM:-0}" = "1" ] && HARNESS_PYTHONPATH="$REPO/scripts/local_cpu_shim"

git -C "$REPO" rev-parse HEAD > "$RUN/commit.txt"
"$PY" - "$REPO" > "$RUN/run_config.json" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
from dataclasses import asdict
from fdagent.voice.livekit_agent import AgentSettings
print(json.dumps(asdict(AgentSettings.from_env()), indent=2))
EOF

for d in "$DATA"/*/; do   # quoted glob: paths may contain spaces
  name=$(basename "$d"); exid="${name%_*}"; spk="${name##*_}"
  [ -n "$ONLY" ] && [ "$exid" != "$ONLY" ] && continue
  OUT="$RUN/examples/$name"; mkdir -p "$OUT"
  rm -f "$d/result_fdagent.json" "$d/output_fdagent.wav" "$d/input_mono.wav"

  cd "$REPO"
  "$PY" -m fdagent.voice.livekit_agent start > "$OUT/agent.log" 2>&1 &
  AGENT=$!
  for i in $(seq 1 300); do
    grep -q "registered worker" "$OUT/agent.log" 2>/dev/null \
      && [ "$(grep -c 'fdagent prewarm complete' "$OUT/agent.log")" -ge 1 ] && break
    kill -0 $AGENT 2>/dev/null || { echo "agent exited during startup" > "$OUT/infra_error"; break; }
    sleep 1
  done

  cd "$V3"
  PYTHONPATH="$HARNESS_PYTHONPATH" timeout 900 "$PY" run_tool_benchmark.py --provider fdagent \
    --example "$exid" --pid "$spk" > "$OUT/harness.log" 2>&1
  echo "$name harness_rc=$?" >> "$RUN/progress.log"

  kill -INT $AGENT 2>/dev/null; sleep 6; kill $AGENT 2>/dev/null; wait $AGENT 2>/dev/null
  for f in result_fdagent.json output_fdagent.wav; do [ -f "$d$f" ] && mv "$d$f" "$OUT/"; done  # inputs stay clean
  rm -f "$d/input_mono.wav"
done

# Official evaluation scripts read result_<provider>.json from the data folders; copy results back
# for the evaluation, then remove them again so the input folders end unchanged.
for o in "$RUN"/examples/*/; do [ -f "$o/result_fdagent.json" ] && cp "$o/result_fdagent.json" "$DATA/$(basename "$o")/"; done
cd "$V3"
LLM=""
case "${OPENAI_API_KEY:-}" in
  ""|your*|*placeholder*) ;;   # unset or the .env.example placeholder: do not claim the LLM judge
  *) LLM="--use-llm" ;;
esac
"$PY" evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider fdagent --output "$RUN/fdagent_evaluation_report.json" $LLM | tee "$RUN/evaluate_tool_calls.log"
"$PY" evaluate_pass_rate.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider fdagent --output "$RUN/fdagent_pass_rate_report.json" $LLM | tee "$RUN/evaluate_pass_rate.log"
[ -n "$LLM" ] && "$PY" analyze_tool_latency.py --results-dir fdb_v3_data_released --provider fdagent \
  --output "$RUN/fdagent_latency_report.json" | tee "$RUN/analyze_tool_latency.log"
for o in "$RUN"/examples/*/; do rm -f "$DATA/$(basename "$o")/result_fdagent.json" "$DATA/$(basename "$o")/latency_tool_analysis_fdagent.json"; done
rm -f "$DATA/evaluation_summary_fdagent.json"
[ -z "$LLM" ] && echo "NOTE: OPENAI_API_KEY not set -> exact-match fallback only (unofficial scores)." | tee -a "$RUN/progress.log"
echo "results: $RUN"
