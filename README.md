# fdagent — an interruptible real-time voice agent (Theme 05, FDB-v3)

**Async intelligence, deterministic state.** A LiveKit voice agent in which every model and tool
runs asynchronously, while a single deterministic *session kernel* is the only component allowed to
change conversation state, dispatch tools, cancel/supersede work, admit results, and decide what is
spoken. Built for the Full-Duplex-Bench v3 (FDB-v3) tool-calling benchmark.

> Status: see [RESULTS.md](RESULTS.md). Results are **local and unofficial**: exact-match scoring
> without the official gpt-4o judge, and the 100-recording run was **stopped after 34 recordings**
> because of the submission deadline.

## Architecture (one diagram)

```
LiveKit room audio
      │
      ▼
 Silero VAD ──► user speech start/end ─────────────────────────┐
      │                                                        │
      ▼                                                        ▼
 STT adapter ──IPC──► dedicated STT process (Whisper base.en)   INGRESS BRIDGE ──► ORDERED INBOX
 (reports started / final / failed per segment)                                     │ (seq = order of truth)
                                                                                    ▼
                                          ┌──────────────────── SESSION KERNEL (single writer) ─────────────────┐
                                          │ fast path · generations · stability + pending-transcript gate       │
                                          │ desired-call keys · reconcile (supersede / reuse / revive / retry)  │
                                          │ result admission · completion guard · stale-speech discard          │
                                          └───────┬───────────────────┬──────────────────────┬──────────────────┘
                              RequestReasoning    │     DispatchTool  │            Speak      │
                                                  ▼                   ▼                       ▼
                                   Gemini planner (JSON plan only;   tool executor ──►    speech channel ──► Piper TTS
                                   never executes tools)             FDB mock tools        (streamed, barge-in aware) ──► LiveKit
                                                  │                   │
                                                  └── proposals / results / timers return to the INBOX ──┘
                                   JOURNAL: every event, decision and action (JSONL, deterministic replay)
```

The full design, invariants and state model are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the original diagram is `01_master_architecture.png`. Code map:

| Concern | Module |
|---|---|
| Kernel, events/actions, gate, reconcile, admission, journal, replay | `fdagent/core/` |
| Runtime loop, tool executor, timers, speech channel | `fdagent/runtime/` |
| FDB-v3 tool specs + official mock backend + telemetry; argument normalization | `fdagent/adapters/` |
| Gemini planner (OpenAI-compatible endpoint), Gemini TTS (optional), Piper TTS (default), OpenAI planner (optional) | `fdagent/providers/` |
| LiveKit entry point, ingress, speech sink, local Whisper STT + STT process, prewarm/job registry | `fdagent/voice/` |

Key guarantees (each covered by tests): slow models/tools never block the conversation; a
self-correction supersedes the old intent and its stale results are never used; a state-changing
call is never executed twice; tool dispatch waits until the user is silent **and** every ended
speech segment has been transcribed; "done" is never spoken without an admitted successful result;
queued speech from an older intent is discarded; each conversation starts from a fresh session.

## Model / provider declaration (zero-cost stack, default `FDAGENT_STACK=gemini_local`)

| Role | Default | Where it runs | Cost |
|---|---|---|---|
| Voice activity detection | Silero VAD (LiveKit plugin) | local CPU | free |
| Speech-to-text | `openai/whisper-base.en` via Hugging Face `transformers` | local CPU, dedicated child process | free |
| Planner | `gemini-3.5-flash-lite` (Google AI Studio free tier, OpenAI-compatible endpoint) | Google API | free tier, no billing |
| Text-to-speech | Piper `en_US-lessac-medium` | local CPU | free |
| Transport | LiveKit Cloud (Build plan) | cloud | free tier |

Optional alternatives: `FDAGENT_STACK=openai` (whisper-1 / gpt-4o / tts-1; needs `OPENAI_API_KEY`),
`FDAGENT_TTS_BACKEND=gemini`, other Whisper sizes via `FDAGENT_WHISPER_MODEL`.

## Setup (Linux / WSL2; Python 3.12; ffmpeg)

```bash
sudo apt install -y ffmpeg python3-venv
python3 -m venv ~/fdagent-venv && . ~/fdagent-venv/bin/activate
pip install -r requirements.txt
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build on GPU machines
pip install -r requirements-local-stt.txt                                     # transformers + piper-tts
bash scripts/fetch_fdb.sh            # FDB-v3 at the pinned commit (+ download the audio as its README says)
bash scripts/fetch_piper_voice.sh    # local TTS voice (~63 MB)
cp .env.example .env.local           # then fill in the values below
```

`.env.local` (never committed; see `.env.example`):

| Variable | Needed for | Where to get it |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | all inference | https://cloud.livekit.io (free Build plan) |
| `GOOGLE_API_KEY` | Gemini planner (default stack) | https://aistudio.google.com (free tier, no billing) |
| `OPENAI_API_KEY` | only the optional `openai` stack, and the official FDB-v3 `--use-llm` judge | OpenAI (paid) |

All other settings (`FDAGENT_*`) have safe defaults documented in `.env.example`.

## Run the agent locally

```bash
python -m fdagent.voice.livekit_agent dev     # connect to LiveKit Cloud; talk to it from the LiveKit Agents Playground
python -m fdagent.voice.livekit_agent start   # production worker (used for the benchmark)
```

Each room gets a fresh kernel and a JSONL journal in `results/journals/<room>.jsonl`; replay one with
`python -m fdagent.core.replay results/journals/<room>.jsonl`.

## Run the benchmark (official FDB-v3 harness, unmodified)

One command (see the script header for options):

```bash
bash scripts/reproduce.sh                       # all 100 recordings
bash scripts/reproduce.sh --example travel_01   # a single scenario
```

It starts a fresh agent per recording, runs `third_party/Full-Duplex-Bench/v3/run_tool_benchmark.py
--provider fdagent`, archives every output under `results/runs/<run_id>/`, then runs the official
evaluation scripts. With `OPENAI_API_KEY` set it runs them with `--use-llm` (the official gpt-4o
judge, paid); without it, it runs the harness's exact-match fallback (unofficial).

CPU-only machines: the harness's `load_asr_model()` calls `.cuda()` unconditionally. Set
`FDAGENT_CPU_HARNESS_SHIM=1` to load `scripts/local_cpu_shim/sitecustomize.py` into the harness
process only (it makes that `.cuda()` a no-op when CUDA is unavailable; it modifies no harness file).
The official evaluation machine has a GPU and does not need it.

## Tests

```bash
python -m pytest -q                                            # unit, race, adapter, integrity tests
FDAGENT_RUN_STT_REGRESSION=1 python -m pytest -q tests/test_stt_regression.py   # needs local models
```

`tests/test_integrity.py` fails if benchmark scenario IDs or expected argument values appear in agent
code or tests.

## Known limitations

- **CPU-only STT latency**: Whisper `base.en` on CPU needs ~2.5–3 s for a 7 s utterance and 7–8 s for
  long (11–15 s) single turns, which dominates perceived latency on this hardware.
- **STT accuracy** on names/IDs over WebRTC audio (e.g. letter O vs digit 0 in order IDs, rare city
  names) is the largest source of wrong arguments.
- **Free-tier planner latency spikes**: occasional 7–19 s Gemini responses (HTTP 200, no retries).
- **Evaluation**: our reported pass counts use the harness's exact-match fallback, not the official
  gpt-4o judge (paid; not run). Exact match is stricter on formatting than the judge.
- **Partial benchmark run**: 34 of 100 recordings completed locally before the deadline.
- **Harness client aborts**: the official `livekit_inference.py` occasionally exits with SIGABRT
  (-6) on this machine; those recordings are reported as infrastructure failures.
- No in-car/extension use case is included in this submission (see SUBMISSION_CHECKLIST.md).

## Licenses / third-party

- `piper-tts` 1.8.0 is **GPL-3.0-or-later**; it is used as an unmodified, separately installed
  dependency (not vendored). The Piper voice is downloaded from `rhasspy/piper-voices`.
- Whisper checkpoints (MIT), LiveKit Agents (Apache-2.0), FDB-v3 harness (fetched, not vendored).
- The FDB-v3 benchmark audio is downloaded separately and is not part of this repository.
