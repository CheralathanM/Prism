# Demo guide (3–5 min video)

No custom UI: the live part uses the LiveKit Agents Playground, the extension part is a console demo.
Everything below runs on the zero-cost stack (local Whisper STT, local Piper TTS, Gemini free-tier
planner, LiveKit Cloud Build plan).

## 0. Before recording

```bash
. ~/fdagent-venv/bin/activate            # Linux/WSL (Windows: .venv\Scripts\activate)
python -m fdagent.voice.livekit_agent dev
```

Open https://agents-playground.livekit.io, connect to the same LiveKit project, and wait for the
agent to join the room (models are prewarmed before the job starts). The demo agent exposes the
FDB-v3 tool set with the in-process mock backend.

## 1. Normal request → tool execution → final spoken answer (live, ~1 min)

Say an ordinary request that needs one tool (any product/flight/account question you like).
Show:

- the agent's short progress line while the tool runs ("Okay, one moment." is only spoken if the
  work takes long),
- the tool call in `/tmp/agent_tool_calls.log` (one line, dispatched exactly once),
- the final spoken answer, which is only produced after the tool **succeeded**.

## 2. User correction + barge-in (live, ~1 min)

Start a request, then correct yourself mid-sentence ("…on Friday — no, sorry, Saturday"), and talk
over the agent while it is answering. Show:

- the agent stops talking immediately (barge-in → `StopSpeaking`),
- only the corrected value is dispatched (the earlier plan is superseded, never executed),
- speech queued for the older turn is discarded, not played late.

Then open the room journal and replay it:

```bash
python -m fdagent.core.replay results/journals/<room>.jsonl     # prints "OK — replay identical"
```

Point out the `barge_in`, `superseded` and `result_rejected` decisions in the JSONL.

## 3. Extension: in-car destination change (console, ~1 min)

```bash
python -m fdagent.extensions.incar.demo --journal results/incar_demo.jsonl
python -m fdagent.core.replay results/incar_demo.jsonl
```

Expected output (scripted, deterministic planner; timings approximate):

```text
[ 0.00s] DRIVER says: Take me to the museum, please.
[ 0.31s] AGENT says: Okay, planning a route to the museum.
[ 0.94s] DRIVER says: Actually, go to the harbour instead.
[ 0.94s] AGENT interrupted (barge-in)
[ 1.31s] AGENT says: Navigating to the harbour. You'll arrive in about 19 minutes.

tool calls executed : plan_route(museum), plan_route(harbour), start_navigation(R-harbour)
navigation started  : ['harbour']  (exactly once, corrected destination only)
superseded ops      : 1   stale results rejected: 1
barge-ins           : 1
```

What it proves: the same kernel, tool protocol, supersession, result admission and speech channel
handle a new domain with **no core changes**. Only two tool specs, a mock backend and a scripted
planner were added. The slow museum route finishes *after* the correction; its result is rejected,
so navigation to the museum never starts.

Variants:

- `--planner gemini` uses the real Gemini planner (free tier, `GOOGLE_API_KEY`) instead of the scripted one.
- `--speak-dir out/` also renders each utterance to WAV with the local Piper voice.

## 4. Close (~30 s)

Show `python -m pytest -q` (172 passed, 2 skipped) and the RESULTS.md headline, which is partial and
unofficial (see the deck).
