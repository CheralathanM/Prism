---
marp: true
title: fdagent — interruptible real-time voice agent
paginate: true
---

# 1 · Problem

**Voice agents that call tools break when people talk like people.**

- Users interrupt, hesitate, correct themselves ("Friday — no, Saturday") and change their minds
  while the agent is still working.
- A naive pipeline (STT → LLM → tools → TTS) executes the *first* version of the request, speaks
  stale answers and duplicates side effects.
- Target: Full-Duplex-Bench v3 (FDB-v3). It uses real spoken, disfluent requests, and a request
  passes only if the agent makes the *right* tool calls.

---

# 2 · Key challenge: full-duplex means everything races

- The user can speak **at any moment**, including over the agent and during tool execution.
- STT, planner, tools and TTS are slow and asynchronous. Their results arrive **out of order and late**.
- State-changing tools (bookings, orders, navigation) must run **at most once, with the latest intent**.
- Constraints we added: **zero API cost** (local Whisper STT, local Piper TTS, Gemini free-tier
  planner, LiveKit Cloud free plan), and **no benchmark knowledge** in code or prompts. An integrity
  test enforces this.

---

# 3 · Architecture: async intelligence, deterministic state

```
mic ─► VAD ─► STT process (Whisper) ─┐
                                      ▼
            ordered inbox ─► SESSION KERNEL (single writer, pure step(event) → actions)
                                      │  ▲            journal (JSONL) → deterministic replay
          ┌───────────┬──────────────┘  │ results (admitted or rejected)
          ▼           ▼                  │
   planner worker   tool executor ───────┘      speech channel ─► Piper TTS ─► LiveKit
   (Gemini: JSON plan only, never executes tools)
```

- Workers only **propose**. The kernel alone changes state, dispatches, cancels, admits results
  and decides what is spoken.
- The kernel does no I/O, reads no clock and uses no randomness, so every session replays identically
  from its journal.

---

# 4 · Core innovation: reconcile-and-commit gate

- Every user speech onset or final transcript bumps a **generation**. Plans are tagged with the
  generation they were made for.
- The planner returns the *desired set* of calls. The kernel **reconciles** it against the calls
  already in flight: it keeps, **supersedes**, revives or retries each one, using a stable call
  key and an idempotency key.
- **Commit gate**: a call is dispatched only if the user is not speaking, the turn is stable
  (0.6 s), the plan matches the current generation, dependencies are met, the arguments pass the
  schema, and no transcription is still pending.
- **Completion guard**: the agent never says "done" unless every desired call succeeded.

---

# 5 · Race-condition and interruption handling

| Race | What the kernel does |
|---|---|
| User speaks over the agent | barge-in → `StopSpeaking`; queued speech for older generations is discarded |
| Correction while a tool is running | old call **superseded**; its late result is **rejected** at admission |
| Transcript still being produced | gate holds dispatch (pending-transcript tracking, deadline extension) |
| Planner answer for an old turn | dropped (generation mismatch) |
| Retry after timeout | same call_id and idempotency key, so no duplicate side effects |

Covered by kernel scenario tests T1–T10, runtime race tests and journal-replay tests.

---

# 6 · Extension use case: in-car destination change

"Take me to the museum" → the agent starts planning the route and saying so → the driver cuts in:
"Actually, go to the harbour instead."

- The agent stops talking. The museum route is **superseded**, and its late result is **rejected**.
- `start_navigation` runs **exactly once**, for the harbour. The final answer gives the new ETA.
- **No core changes.** The extension adds 2 tool specs, a mock car backend and a scripted planner.
  `--planner gemini` swaps in the real planner.
- `python -m fdagent.extensions.incar.demo`; regression test `tests/test_incar_extension.py`.

---

# 7 · Validation and results (local, unofficial)

- **Tests:** 172 passed, 2 skipped (Windows). The suite covers kernel scenarios, races, adapters,
  integrity checks and the extension. Journal replay is identical.
- **Six-case validation:** pipeline correct **6/6**; exact match **4/6**. Both failures are STT
  errors ("PO999" was heard as "P0999", and "Seoul" as "soil").
- **Partial FDB-v3 run: 34/100 recordings** (stopped at the deadline). *Partial and unofficial*:
  scored by exact match, not the official gpt-4o judge.
  - **Agent:** 17/30 scored recordings pass. Failures: 6 STT, 6 tool/argument, 1 planner.
    0 identical duplicate calls; self-correction 3/3; 15 barge-ins handled.
    Median perceived latency 7.5 s (p90 21.5 s).
  - **Infrastructure, excluded:** 4 failures (3 harness-client aborts, exit −6; 1 host standby).

---

# 8 · Demo, limitations and future work

**Demo:** a live LiveKit Playground session (normal request, correction, barge-in) plus the in-car
console demo and a journal replay. See `docs/DEMO.md`.

**Limitations**
- CPU Whisper `base.en` is the bottleneck. It mishears IDs and rare names and takes 2.5–8 s per turn.
- The free-tier planner makes occasional wrong or extra tool choices and has 7–19 s latency spikes.
- Results cover only 34/100 recordings and are not officially judged.

**Future work:** GPU or streaming STT with ID-aware post-processing; speculative read-only
dispatch; the full 100-recording run with the official judge; more in-car tools (stops, reroute
on traffic).
