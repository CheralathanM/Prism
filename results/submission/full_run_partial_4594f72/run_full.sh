#!/usr/bin/env bash
# Full FDB-v3 run (100 recordings) at commit 4594f72, official runner, unmodified harness/scorer/data.
# One fresh agent per recording (--example <id> --pid <speaker>). LOCAL CPU-ONLY harness shim on the
# harness process only. All generated outputs are moved out of the example folders into $RUN/examples.
# Resumable: a recording with a "done" marker is not rerun.
set -uo pipefail
REPO="/mnt/c/Users/chera/Downloads/Samsung Prism"
V3="$REPO/third_party/Full-Duplex-Bench/v3"
DATA="$V3/fdb_v3_data_released"
RUN="$REPO/results/local_smoke/full_run_4594f72"
PY=~/fdagent-venv/bin/python
PROGRESS="$RUN/progress.log"
mkdir -p "$RUN/examples"

export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
set -a; . "$REPO/.env.local"; set +a   # credentials into env; never echoed or written
unset OPENAI_API_KEY
export FDAGENT_STACK=gemini_local PYTHONUNBUFFERED=1

if [ ! -f "$RUN/run_config.json" ]; then
  "$PY" - "$REPO" > "$RUN/run_config.json" <<'EOF'
import json, subprocess, sys, platform, datetime, importlib.metadata as md
repo = sys.argv[1]
sys.path.insert(0, repo)
from dataclasses import asdict
from fdagent.voice.livekit_agent import AgentSettings
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
pk = {}
for p in ("livekit-agents", "livekit", "openai", "httpx", "torch", "transformers", "piper-tts", "onnxruntime",
          "nemo-toolkit", "numpy"):
    try:
        pk[p] = md.version(p)
    except md.PackageNotFoundError:
        pk[p] = None
print(json.dumps({
    "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
    "fdb_commit": git("-C", "third_party/Full-Duplex-Bench", "rev-parse", "HEAD"),
    "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "settings": asdict(AgentSettings.from_env()),
    "packages": pk, "python": platform.python_version(), "platform": platform.platform(),
    "harness": "run_tool_benchmark.py --provider fdagent --example <id> --pid <speaker> (one recording per agent)",
    "harness_workaround": "LOCAL CPU-ONLY shim: lightning DeviceDtypeModuleMixin.cuda() returns self when CUDA is unavailable",
    "scoring_note": "exact-match via evaluate_scenario_pass(use_llm=False) is unofficial; the official gpt-4o judge was not run",
}, indent=2))
EOF
  cp "$(dirname "$0")/cpu_shim/sitecustomize.py" "$RUN/cpu_shim/" 2>/dev/null || true
fi

for d in "$DATA"/*/; do   # quoted glob: the repo path contains a space ("Samsung Prism")
  name=$(basename "$d"); exid="${name%_*}"; spk="${name##*_}"
  OUT="$RUN/examples/$name"
  [ -f "$OUT/done" ] && continue
  mkdir -p "$OUT"
  rm -f "$d/result_fdagent.json" "$d/output_fdagent.wav" "$d/input_mono.wav"   # generated files only

  T0=$(date +%s)
  cd "$REPO"
  "$PY" -m fdagent.voice.livekit_agent start > "$OUT/agent.log" 2>&1 &
  AGENT=$!
  ready=0
  for i in $(seq 1 300); do
    if [ -f "$OUT/agent.log" ] && grep -q "registered worker" "$OUT/agent.log" \
       && [ "$(grep -c 'fdagent prewarm complete' "$OUT/agent.log" 2>/dev/null || echo 0)" -ge 2 ]; then ready=1; break; fi
    kill -0 $AGENT 2>/dev/null || break
    sleep 1
  done
  if [ "$ready" != 1 ]; then echo "agent not ready after ${i}s" > "$OUT/infra_error"; fi

  cd "$V3"
  PYTHONPATH="$RUN/cpu_shim" timeout 900 "$PY" run_tool_benchmark.py --provider fdagent --example "$exid" --pid "$spk" \
    > "$OUT/harness.log" 2>&1
  RC=$?

  sleep 3
  kill -INT $AGENT 2>/dev/null; sleep 6; kill $AGENT 2>/dev/null; wait $AGENT 2>/dev/null
  sleep 2
  ps -eo pid,ppid,cmd | grep -E "fdagent.voice.livekit_agent|multiprocessing.spawn" | grep -v grep > "$OUT/leftover_procs.txt" || true

  for f in result_fdagent.json output_fdagent.wav; do [ -f "$d$f" ] && mv "$d$f" "$OUT/"; done
  rm -f "$d/input_mono.wav"
  [ -f "$DATA/evaluation_summary_fdagent.json" ] && mv "$DATA/evaluation_summary_fdagent.json" "$OUT/"
  ROOM=$("$PY" -c "import json;print(json.load(open('$OUT/result_fdagent.json')).get('room_name',''))" 2>/dev/null || true)
  STATUS=$("$PY" -c "import json;print(json.load(open('$OUT/result_fdagent.json')).get('status',''))" 2>/dev/null || echo none)
  if [ -n "$ROOM" ]; then
    cp "$REPO/results/journals/$ROOM.jsonl" "$OUT/journal.jsonl" 2>/dev/null
    grep -F "\"$ROOM\"" /tmp/agent_tool_calls.log > "$OUT/tool_calls.log" 2>/dev/null
    grep -F "\"$ROOM\"" /tmp/agent_heartbeat.log > "$OUT/heartbeat.log" 2>/dev/null
  fi
  echo "$(date -Is) $name harness_rc=$RC status=$STATUS room=$ROOM ready=$ready secs=$(( $(date +%s) - T0 ))" >> "$PROGRESS"
  touch "$OUT/done"
done
echo "$(date -Is) RUN_COMPLETE" >> "$PROGRESS"
