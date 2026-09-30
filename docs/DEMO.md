# Demo guide (3–5 min video)

The whole recommended video runs on one laptop with **no API keys**. It uses the browser demo UI
over the real kernel and runtime, and the scripted planner is deterministic. An optional live-voice
segment uses the LiveKit agent on the zero-cost stack (local Whisper STT, local Piper TTS, Gemini
free-tier planner, LiveKit Cloud Build plan).

## 0. Before recording

```bash
python -m fdagent.extensions.incar.ui        # then open http://127.0.0.1:8765 (full screen, zoom ~110 %)
```

Tick **Voice** if you want the agent's lines spoken aloud; the browser voices them locally.

**Speaking instead of clicking (🎤 Mic, Chrome or Edge):** click **🎤 Mic off**, allow the microphone,
and say "Take me to the museum". While the agent is answering, say "Actually, go to the harbour
instead". Wear **headphones** if Voice is on, otherwise the mic picks up the agent's voice and
interrupts it. If the page shows an error after clicking Mic, restart the UI server so it has the
latest code.
Do one dry run first (**Run demo**), then press **Reset session**.

## 1. Recommended 3–5 minute sequence

| Time | Show | Say / do |
|---|---|---|
| 0:00–0:40 | Architecture slide (deck slide 3), then the UI | "Every model and tool runs asynchronously, but one deterministic kernel is the only thing allowed to change state. This page is just a live view of that kernel's journal." Point at the four areas. |
| 0:40–1:00 | **Say: “Take me to the museum”** | Agent state goes **Planning**, then **Speaking**. The tool log shows `plan_route(destination="museum") → dispatched`, plus a gate hold on `start_navigation` (waiting for its dependency). |
| 1:00–1:15 | Let the agent start "Okay, planning a route to the museum." | The museum route is slow (5 s mock latency), so it is still running. |
| 1:15–1:30 | **Interrupt: “Actually, go to the harbour instead”** while it is speaking | The state flips to **Interrupted** and the agent's line is struck through as INTERRUPTED. The interruption panel turns red. |
| 1:30–2:00 | Interruption panel | Read the steps: speech stopped → new intent admitted (generation, correction cue "actually") → previous response cancelled → **old action superseded**. |
| 2:00–2:30 | Tool log | `plan_route(museum)`: superseded, cancel requested. `plan_route(harbour)`: completed. `start_navigation(route_id="R-harbour")`: navigation started. A few seconds later the museum route finishes anyway and shows **late result rejected**; it is never applied. |
| 2:30–2:50 | Vehicle card and conversation | "Navigating to the harbour · ETA … · navigation starts: **1**". Final agent line: "Navigating to the harbour. You'll arrive in about … minutes." State: **Completed**. |
| 2:50–3:20 | **Replay last journal** | The banner reads "… kernel steps re-executed, decisions and actions identical ✓", and the whole session re-renders from the journal. |
| 3:20–4:30 | Optional: live voice (section 2), or the results slide | Only verified numbers: 34/100 partial and unofficial; infrastructure failures listed separately. |
| 4:30–5:00 | Limitations / future work slide | |

If you are late with the interrupt, the flow is still correct but less visual: no barge-in, but the
museum route is still superseded, provided it was still running. Press **Reset session** and redo
it, or just use **Run demo**, which interrupts automatically 1.2 s into the agent's sentence.

## 2. Optional: live voice with the LiveKit agent (FDB-v3 tools)

```bash
python -m fdagent.voice.livekit_agent dev
```

Open https://agents-playground.livekit.io, connect to the same LiveKit project, and talk to the
agent. It uses the FDB-v3 tool set with the in-process mock backend.

- Ask an ordinary one-tool request. Tool calls appear in `/tmp/agent_tool_calls.log`, and the final
  answer comes only after the tool succeeded.
- Correct yourself mid-sentence ("…Friday — no, Saturday") and talk over the agent. It stops
  immediately, and only the corrected value is dispatched.
- Replay the room journal:

  ```bash
  python -m fdagent.core.replay results/journals/<room>.jsonl
  ```

## 3. Console variant of the extension (no browser)

```bash
python -m fdagent.extensions.incar.demo --journal results/incar_demo.jsonl
python -m fdagent.core.replay results/incar_demo.jsonl
```

Prints the same museum → harbour flow: navigation started exactly once (harbour), 1 superseded
operation, 1 stale result rejected, 1 barge-in.

## What the demo proves

- The same kernel, tool protocol, supersession, result admission and speech channel handle a new
  domain with **no core changes**. The extension adds two tool specs, a mock backend, a scripted
  planner and a view.
- The UI shows the real journal. Nothing on screen is hard-coded, and a replay renders exactly the
  same events.
