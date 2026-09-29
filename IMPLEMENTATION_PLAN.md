# Implementation Plan — Interruptible Real-Time Agent (Theme 05, FDB-v3)

Core principle: **async intelligence, deterministic state.**
Architecture source: `01_master_architecture.png` + the design brief. This plan maps it to code.

---

## 1. Current state (Phase 0 audit, 2026-09-29)

| Item | Status |
|---|---|
| Project code | **None.** Folder contained only `Theme05_Participant_Guide_UPDATED_FBD.docx` and `01_master_architecture.png`. |
| Git | Not a repository yet. |
| FDB-v3 | Cloned to `third_party/Full-Duplex-Bench` at commit `3e799c45a045256f47d5f1c9cda90157e2d2ec9e` (2026-05-20). Read and analysed (below). |
| Benchmark audio | **Not downloaded.** Google Drive link in `v3/README.md`; needs a human download. |
| Local toolchain | Python 3.12.10, Node 24, git 2.53. **No** ffmpeg, no NVIDIA GPU / `nvidia-smi`, `livekit-agents` not installed. Windows 11. |
| Credentials | None present. Needed: LiveKit Cloud (URL/key/secret), OpenAI API key (the official judge is gpt-4o, so the key is required anyway). |

### 1.1 Verified facts about the FDB-v3 harness (these drive the design)

Everything below is read from the pinned commit, not assumed.

1. **Flow.** `run_tool_benchmark_all_released.py` → for each `fdb_v3_data_released/{example}_{speaker}/input.wav` it runs `livekit_inference.py`, which joins room `eval-<uuid8>` as a "user", streams the WAV in real time (20 ms frames, 48 kHz) plus 1.5 s trailing silence, and records the agent's audio. **The recording window equals the input duration.** Agent speech after that is lost.
2. **Tool calls are scored from the agent's own telemetry.** The harness reads `/tmp/agent_tool_calls.log` (JSONL: `{"room", "call": {"function", "args", "timestamp_start", "timestamp_end"}}`) and keeps the lines whose `room` matches. Our agent must write this file in exactly that format.
3. **Strict pass = exact multiset of tools + all arguments correct** (`evaluate_pass_rate.py`). **Any extra executed call fails the scenario**, read-only ones included. A premature `search_flights(Rome)` before the user says "no wait, Milan" costs the whole scenario even if the Milan call follows. ⇒ **The commit gate must hold every tool, not only state-changing ones, until intent is stable.** This is the single most important benchmark-driven decision.
4. Argument matching uses a gpt-4o judge with lenient rules ("July 15" == "2026-07-15"). `$RESULT_0.x` references accept any plausible value from a previous result.
5. Latency: first response and tool-call latency are measured from the *end of the first user turn*, detected as the first ASR word gap > 2 s in `input.wav`. Task-completion latency is judged by gpt-4o on the output transcript.
6. Data: 100 examples / 79 scenarios, every scenario is **2 user turns in one recording** (turn 1 is usually small talk). Features: FILLER 26, SELF_CORRECTION 21, PAUSE 19, FALSE_START 11, HESITATION 10; 21 `state_rollback_test`; difficulty easy 32 / medium 36 / hard 32.
7. `mock_apis.py` tool bodies are deterministic, **but** `latency_injector.py` applies per-API overrides even under the `instant` profile (`search_flights`, `search_apartments`, `calculate_commute` → 200–800 ms; `update_search_filter` → 50–200 ms), uses an **unseeded** `random.randint`, and sleeps with blocking `time.sleep`. In the templates that sleep runs inside `async def` tools, **blocking the agent's event loop**, a real instance of the latency problem the challenge describes. We run mock calls in a worker thread and seed the RNG.
8. ASR in the harness is NVIDIA NeMo `parakeet-tdt-0.6b-v2` (GPU, Linux). Paths are hardcoded to `/tmp`. **Evaluation must run on Linux** (the official re-run machine is Linux + 48 GB GPU). Locally, that means WSL2 or a cloud box.

---

## 2. Architecture → code mapping

| Architecture box | Module | Responsibility |
|---|---|---|
| Protocol adapter · ingress | `fdagent/voice/livekit_agent.py` (+ `fdagent/core/events.py`) | Only code that knows LiveKit/STT wire formats; converts VAD/STT callbacks into `Event`s. |
| Inbox | `fdagent/core/inbox.py` | Single ordered queue. `seq` assigned at enqueue = order of truth. Clock injected. |
| Session kernel | `fdagent/core/kernel.py` | **Single writer.** One synchronous `step(envelope) -> [Action]` per event. Never awaits models or tools. |
| Fast understanding | `fdagent/core/fast_path.py` | Deterministic, in-step: correction-cue detection, barge-in, backchannel text. |
| Session state | `fdagent/core/state.py`, `fdagent/core/model.py` | Generation counter, transcript, desired calls (by key), operation ledger, admitted facts, tool registry. |
| Reconcile | `fdagent/core/reconcile.py` | Desired vs actual: new key → dispatch candidate; same key → keep/reuse; key gone → SUPERSEDED + advisory cancel. |
| State-changing gates / commit barrier | `fdagent/core/gate.py` | Stability, freshness, dependency, and schema checks. One unresolved attempt per operation, never a second create. |
| Result admission | `fdagent/core/admission.py` | Every outcome is recorded as fact; used only if the call is still desired and the attempt is current. |
| Speech + snapshot | `fdagent/core/kernel.py` (`_speak_*`), `Kernel.snapshot()` | Speech derived from state. No "done" without an admitted success. |
| Slow path: reasoner / planner | `fdagent/runtime/reasoner.py` | LLM worker. Gets a read-only snapshot and returns a `ReasonerProposal` event. Never touches state. |
| Slow path: perception | STT inside LiveKit session (ingress) | Speech-to-text evidence arrives as `UserTranscript` events. (Camera perception: out of scope unless the extension needs it.) |
| Slow path: timers | `fdagent/runtime/timers.py` | Stability windows, tool timeouts. Fire back as `TimerFired` events. |
| Protocol adapter · egress / tool executor | `fdagent/runtime/tool_executor.py`, `fdagent/adapters/*` | Schema-validate, execute in a thread, pass the idempotency key, write FDB telemetry, return `ToolResult` events. |
| Mock/benchmark tools | `fdagent/adapters/fdb_tools.py` → `third_party/.../v3/mock_apis.py` | 12 FDB tool specs bound to the official mock backend. |
| Journal | `fdagent/core/journal.py`, `fdagent/core/replay.py` | JSONL of every envelope, decision, and action. Offline deterministic replay. |
| Runtime loop | `fdagent/runtime/loop.py` | Drains inbox → `kernel.step` → executes actions as asyncio tasks. |

## 3. Target project tree

```
Samsung Prism/
├── IMPLEMENTATION_PLAN.md   README.md   SUBMISSION_CHECKLIST.md
├── pyproject.toml           .env.example   .gitignore
├── fdagent/
│   ├── core/        ids, events, actions, model, state, inbox, config,
│   │                fast_path, reconcile, gate, admission, kernel, journal, replay
│   ├── runtime/     loop, tool_executor, reasoner, timers
│   ├── adapters/    tool_protocol, fdb_tools, car_tools (extension)
│   └── voice/       livekit_agent (FDB entry point), car_agent (extension)
├── tests/           unit + deterministic race tests (no network)
├── scripts/         setup.sh, fetch_fdb.sh, run_benchmark.sh, evaluate.sh, reproduce.sh
├── results/         <run_id>/ reports, journals, config+seed record
├── docs/            architecture diagram (source + png), extension.md, limitations.md
└── third_party/Full-Duplex-Bench   (pinned commit, gitignored, fetched by script)
```

## 4. Data flow (one user turn)

```
VAD start ─► UserSpeechStarted ─► kernel: generation++, barge-in → StopSpeaking
STT final ─► UserTranscript    ─► kernel: generation++, correction cue noted, RequestReasoning(req, gen)
VAD end   ─► UserTurnEnded     ─► kernel: StartTimer(stability, gen)
timer     ─► TimerFired(gen)   ─► kernel: if gen still current & user silent → stable; backchannel Speak
reasoner  ─► ReasonerProposal  ─► kernel: accept only if req is latest & gen current → desired := calls
                                   reconcile → gate → DispatchTool(call_id, attempt, key, idem_key)
executor  ─► ToolResult        ─► admission → fact; reconcile (next chain step unblocks) → RequestReasoning
reasoner  ─► ReasonerProposal(reply) ─► completion guard → Speak(final) only if every desired call SUCCEEDED
```

## 5. State model

- `generation: int` bumps on every new user speech onset and every final transcript segment. Everything asynchronous carries the generation it was based on.
- `stable: bool`: the user stopped speaking and the stability timer for the *current* generation fired.
- `desired: dict[key, DesiredCall]`: the complete set of calls the current intent requires (from the latest admitted proposal). Each entry records its generation.
- `ops: dict[call_id, Operation]`, `op_by_key: dict[key, call_id]`: operation ledger.
- **Desired call key** = `tool` + canonical(args, case/whitespace/number-normalised; dependency refs rendered as `$ref:<dep key>/path`) + `occurrence`. Identical calls collapse unless the planner explicitly asks for a repeat via `occurrence`.
- `facts`: results admitted for desired keys. `orphan_effects`: state-changing results that arrived after their op was superseded. They are recorded, never used, and reported honestly.

## 6. Event types

`UserSpeechStarted`, `UserTranscript(text, final)`, `UserTurnEnded`, `AgentSpeechStarted`, `AgentSpeechEnded`, `ReasonerProposal(request_id, generation, calls, reply, reply_kind)`, `ReasonerFailed`, `ToolResult(call_id, attempt, ok, payload, error, retriable, effect_applied)`, `TimerFired(timer_id, kind, generation, call_id, attempt)`.

Actions: `Speak`, `StopSpeaking`, `RequestReasoning`, `DispatchTool`, `AdvisoryCancel`, `StartTimer`.

## 7. Operation lifecycle

```
(desired, gated) ──gate ok──► DISPATCHED ──ok──────► SUCCEEDED   (terminal; key reused, never re-created)
                                 │        ──error───► FAILED      (retry same call_id/idempotency key if retriable & attempts left)
                                 │        ──timeout─► UNKNOWN     (read-only: retry; state-changing: never auto-retry)
                                 └─key no longer desired─► SUPERSEDED (advisory cancel; late result logged, never used;
                                                                      if the key becomes desired again the op is revived, not duplicated)
```

## 8. Gate rules (commit barrier). A desired call is dispatched only if all of these hold:
1. The user is not speaking and `stable` is true for the current generation (holds **all** tools, per §1.1-3).
2. The desired call's generation equals the current generation (no dispatching an old plan while a fresh one is being computed).
3. All dependencies have SUCCEEDED, so refs resolve.
4. The tool is known and its args pass the schema.
5. No existing op for the key is DISPATCHED/SUCCEEDED, and for state-changing tools none is UNKNOWN.

## 9. Test plan

`tests/` covers deterministic, network-free kernel tests for the 10 mandated scenarios, each naming its invariant:
T1 argument change mid-flight · T2 stale result after new intent · T3 duplicate state-changing request · T4 retry after partial completion (idempotency key) · T5 three-step chain · T6 interruption during chain step 2 · T7 many events while reasoner is slow · T8 late stale result after success · T9 failure means no completion claim · T10 new intent obsoletes old desired action.
Plus: journal replay equivalence, the gate holding during speech, and stale-timer rejection. Phase 2 adds asyncio race tests with the real runtime loop and fake tools. Phase 3 adds a LiveKit smoke test (manual, logged).

## 10. Benchmark integration plan

1. `scripts/fetch_fdb.sh`: clone the pinned commit and verify the data folder exists.
2. `fdagent/voice/livekit_agent.py` is a LiveKit `AgentServer` entry point (same pattern as `cascaded_agent.py`), started with `python -m fdagent.voice.livekit_agent start`. It writes `/tmp/agent_tool_calls.log` and `/tmp/agent_heartbeat.log` in the official format.
3. Run the official, unmodified `run_tool_benchmark_all_released.py --provider fdagent`, then `evaluate_tool_calls.py`, `evaluate_pass_rate.py`, and `analyze_tool_latency.py` with `--use-llm`.
4. `scripts/reproduce.sh` does install → configure check → start agent in background → inference → evaluation → copy reports + journals + config/seed into `results/<run_id>/`.
5. First milestone: **one scenario end-to-end** (`--example travel_01`). Then run all 100.
6. Integrity: no scenario IDs, expected args, or transcripts appear in agent code; a test greps for that. There is no cross-room state: one kernel per LiveKit room/job.

## 11. Extension plan

**In-car destination change** (`fdagent/voice/car_agent.py` + `adapters/car_tools.py`): `plan_route`, `start_navigation` (state-changing), and `send_eta` (state-changing). "Take me to the airport… actually, the railway station" must end with exactly one `start_navigation(railway station)`, plus a demo of a late airport route result being rejected. It is the same kernel with a different tool registry. It runs in LiveKit's playground/console, and the journal is shown in the video.

## 12. Decisions and open questions

- **Model stack (proposed, to confirm in Phase 3):** cascaded pipeline in LiveKit. Silero VAD, streaming STT, and TTS stay in the LiveKit session. An OpenAI chat model with tool-calling is the *reasoner* only. We use one provider (OpenAI) so evaluators need a single key besides LiveKit, which they need anyway for the gpt-4o judge. Exact model IDs will be pinned after testing.
- **Reasoner contract:** each proposal is the *complete* desired call set for the current generation. The reasoner worker re-includes still-valid executed calls, so the kernel can supersede anything omitted.
- **Speculative reads:** off for the benchmark (§1.1-3). A config flag `gate_read_only` allows the extension to speculate if we want.
- **Local evaluation on Windows:** not possible as-is (`/tmp`, NeMo GPU ASR). Use WSL2/Linux for Phases 4–8.

## 13. First five milestones

1. **Core kernel + journal + tests T1–T10** (pure Python, no network) — *started in this pass.*
2. Runtime loop, timers, tool executor, and FDB tool adapter, with asyncio race tests using fake tools.
3. OpenAI reasoner worker plus an offline text-driven simulator. This is a dev-only harness using *our own* synthetic dialogues, not FDB items.
4. LiveKit agent wrapper: one real FDB scenario end-to-end on Linux, with latency instrumentation.
5. Full 100-example FDB run, `reproduce.sh`, results folder, then failure analysis.
