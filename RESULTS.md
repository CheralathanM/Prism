# Results — fdagent on FDB-v3 (local, unofficial)

**Commit used for all results below: `4594f72`** (clean tree).
Stack: Silero VAD · Whisper `base.en` (local CPU, dedicated STT process) · Gemini `gemini-3.5-flash-lite`
planner (free tier) · Piper TTS (local) · LiveKit Cloud. Harness: official FDB-v3
`run_tool_benchmark.py --provider fdagent`, unmodified, one fresh agent per recording.

> **Read this first**
> - **Not an official score.** Pass/fail below uses the harness's own `evaluate_scenario_pass` with
>   `use_llm=False` (exact-match fallback). The official gpt-4o judge (`--use-llm`) is paid and was not
>   run; it accepts formatting variants that exact match rejects.
> - **Not a complete benchmark.** The 100-recording run was stopped deliberately at the deadline after
>   **34 completed recordings** (alphabetical order: all `ecommerce_*` and the first `finance_*`).
> - Local machine is CPU-only (no CUDA). The harness's ASR `.cuda()` call was bypassed with a local
>   shim loaded only into the harness process (`scripts/local_cpu_shim/`); no harness file was changed.
> - Infrastructure failures are reported separately and are **not** counted as agent failures.

Raw material: `results/submission/` (per-recording harness results, kernel journals, tool-call and
latency telemetry, harness logs, run config, analysis scripts).

## 1. Six-case validation (validation_3)

Six recordings chosen to exercise self-correction, barge-in, dependent multi-tool calls, small talk,
date normalization and an ordinary request. Fresh agent per case.

| Case | What it tests | Pipeline behaviour | Exact-match | Perceived latency |
|---|---|---|---|---|
| travel_10 | self-correction | only the corrected call dispatched; stale reply dropped | PASS | 12.4 s |
| ecommerce_05 | barge-in, hesitation, split request | barge-in handled; single call **after** all segments transcribed | PASS | 18.0 s |
| ecommerce_18 | 3 dependent tools | all 3 dispatched once, add_to_cart used the search result | FAIL (STT: "PO999" heard as "P0999") | 14.6 s |
| finance_16 | small talk then request | no tool call for small talk; correct call | PASS | 11.7 s |
| travel_14 | date normalization | date normalized ("…22nd" → "March 22"); one call | FAIL (STT: "Seoul" heard as "soil") | 26.0 s (12.6 s planner spike) |
| housing_01 | ordinary request | typed numeric args correct; one call | PASS | 19.0 s |

**Pipeline: 6/6** (each expected tool dispatched exactly once, no premature or duplicate dispatch,
speech captured in the recording, barge-in/stale-speech behaviour correct).
**Exact-match: 4/6** — both failures are speech-recognition errors, not kernel/planner errors.
One earlier attempt of ecommerce_05 ended with the harness's own client aborting (SIGABRT);
our agent handled that session correctly and a plain rerun was clean (both kept in the package).

Before/after the last fix (moving Whisper out of the agent process): event-loop blocking per case
fell from up to 6.8 s to one-time warm-up costs; the ecommerce_05 transcription backlog fell from
30 s to ~1–1.5 s per segment; the duplicate/premature dispatch disappeared.

## 2. Partial full run — 34 of 100 recordings completed

| Metric | Value |
|---|---|
| Recordings completed by the harness | 34 (plus 1 interrupted at the stop, excluded) |
| Infrastructure failures (excluded from scoring) | **4** — see §3 |
| Scored recordings | **30** |
| Exact-match pass (unofficial) | **17 / 30 (56.7 %)** |
| Exact-match fail | 13 / 30 |
| By difficulty | easy 8/12 · medium 4/10 · hard 5/8 |
| By disfluency feature | FALSE_START 4/4 · SELF_CORRECTION 3/3 · FILLER 4/7 · HESITATION 1/3 · PAUSE 1/5 |
| Identical duplicate tool calls | **0** |
| Extra (unexpected) calls | 3, in 3 recordings |
| Recordings with a premature dispatch (same tool re-dispatched with changed args) | 1 |
| Self-correction scenarios (state rollback) | 3/3 executed only the corrected value; 0 stale values executed |
| Barge-in | 15 events in 11 recordings, all handled (speech interrupted); 1 stale queued utterance dropped |
| Agent speech captured in the recording | 25 / 34 |
| Speech not delivered | 6 utterances queued after the harness had left the room ("AgentSession isn't running"); no TTS synthesis failures |
| Perceived latency (harness, n=25) | median **7.5 s** · mean 10.8 s · p90 21.5 s · min 2.6 s · max 28.4 s |
| Gemini planner HTTP attempts | 138, all HTTP 200; 1 SDK retry; **0 × 429 rate limits**; 0 reasoner failures; slowest attempt 19.4 s |
| Transcription safety handling | 137 deadline extensions while STT was busy; 9 safety timeouts; 1 STT failure |

### Failing scored recordings (13), by primary category

Categories are assigned automatically by `results/submission/full_run_partial_4594f72/full_analyze.py`
(e.g. *STT* when a required value is in the scripted utterance but absent from our transcript) and are
indicative, not hand-audited.

| Category | Count | Recordings (reason) |
|---|---|---|
| STT/transcription | 6 | ecommerce_12 (value "gift" not heard) · ecommerce_13 ×2 (order ID "DELIV…" not heard) · ecommerce_18 ("gaming mouse", "PO999") · ecommerce_21 (order ID "BOB…") · finance_01 (amount/currencies not heard; no call) |
| Tool selection / arguments | 6 | ecommerce_06 (extra search_products) · ecommerce_08 (search args) · ecommerce_10 (search args) · ecommerce_14 (add_to_cart args) · ecommerce_16 (search_products twice) · ecommerce_21 (track_order args) |
| Planner | 1 | ecommerce_08 (second speaker: no search_products call) |
| Timing/latency | 0 | — |
| TTS/audio | 0 as primary cause | (6 late utterances refused after the session closed, see above) |

## 3. Harness / infrastructure failures (not agent failures)

| Recording | What happened | Evidence |
|---|---|---|
| ecommerce_04_6998… | official `livekit_inference.py` aborted (exit −6, SIGABRT) → `inference_failed` | harness.log; agent journal shows a normal session |
| ecommerce_14_5ff0… | same harness-client abort (−6) | harness.log |
| ecommerce_15_5678… | same harness-client abort (−6) | harness.log |
| ecommerce_02_5f4a… | host entered Modern Standby during job assignment; LiveKit job assignment timed out, the agent never joined (recording took 1,236 s) | agent.log: "process is unresponsive", "assignment for job … timed out"; Windows Kernel-Power 506/507 events |

The client aborts did not correlate with agent behaviour (agent silent 10–22 s before disconnect in
all three). Standby was prevented for the rest of the run with a keep-awake request.

## 4. Remaining limitations

- **STT accuracy on IDs and rare names** over WebRTC audio (letter O vs digit 0, city names) is the
  largest single cause of failures. `base.en` was chosen over `tiny.en` on a local synthetic regression
  set (9–10/10 vs 8–9/10 key values); larger models were not evaluated on this CPU.
- **CPU-only STT latency**: 2.5–3 s for a 7 s utterance; 7–8 s for long 11–15 s turns.
- **Planner**: occasional wrong/extra tool selection and argument formatting; free-tier latency spikes
  of 7–19 s (HTTP 200, no retries).
- **Scoring**: all numbers are exact-match (unofficial); the official gpt-4o judge was not run.
- **Coverage**: 34/100 recordings; finance/housing/travel domains are mostly unmeasured in the full run.
