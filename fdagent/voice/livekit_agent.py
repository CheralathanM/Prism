#!/usr/bin/env python3
"""FDB-v3 LiveKit agent: LiveKit handles audio I/O; the session kernel handles everything else.

Follows the official template pattern (``v3/cascaded_agent.py``): an ``AgentServer`` with an
``rtc_session`` entrypoint, one session per room, and tool telemetry in the harness format.
Differences:
  * AgentSession runs VAD + STT (+ TTS) only (no LLM): LiveKit's own reply loop is disabled.
  * VAD/STT callbacks become kernel events (ingress); kernel Speak actions become
    ``session.say`` (egress). Tool calls go kernel → executor → official mock backend.
  * The planner only proposes plans; it cannot execute tools.

Provider stacks (``FDAGENT_STACK``):
  gemini_local (default, zero-cost): local Whisper STT, Gemini planner (OpenAI-compatible
               endpoint), Gemini TTS. Needs GOOGLE_API_KEY + LiveKit credentials only.
  openai:      OpenAI whisper-1 STT, gpt-4o planner, tts-1. Needs OPENAI_API_KEY.

Run (from the repo root, with .env.local configured):
    python -m fdagent.voice.livekit_agent start      # or: dev / console
Provider name for the harness: ``--provider fdagent``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, StopResponse

from fdagent.adapters.fdb_tools import FDB_TOOL_SPECS, FdbMockBackend
from fdagent.core.config import KernelConfig
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.providers.gemini_reasoner import DEFAULT_GEMINI_PLANNER_MODEL, DEFAULT_REASONING_EFFORT
from fdagent.providers.gemini_tts import DEFAULT_GEMINI_TTS_MODEL, DEFAULT_GEMINI_VOICE
from fdagent.providers.openai_reasoner import DEFAULT_MODEL
from fdagent.runtime.loop import SessionRuntime
from fdagent.voice.fdb_heartbeat import FdbLatencyLog
from fdagent.voice.ingress import IngressBridge
from fdagent.voice.job_registry import ActiveJobRegistry

log = logging.getLogger("fdagent.livekit")
REPO_ROOT = Path(__file__).resolve().parents[2]
STACKS = ("gemini_local", "openai")
TTS_BACKENDS = ("piper", "gemini")


def _choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value
DEFAULT_WHISPER_MODEL = "openai/whisper-base.en"  # mirrors voice.local_whisper (avoids importing it eagerly)


@dataclass(frozen=True)
class AgentSettings:
    provider: str = "fdagent"
    stack: str = "gemini_local"
    latency_profile: str = "instant"
    # gemini_local stack
    gemini_planner_model: str = DEFAULT_GEMINI_PLANNER_MODEL
    gemini_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    gemini_tts_model: str = DEFAULT_GEMINI_TTS_MODEL
    gemini_voice: str = DEFAULT_GEMINI_VOICE
    tts_backend: str = "piper"  # "piper" (local, default; no quota) | "gemini" (optional, free-tier quota)
    piper_voice: str = ""  # path to a Piper .onnx voice; "" = local_piper_tts.DEFAULT_PIPER_VOICE
    tts_streaming: bool = True
    tts_prebuffer_ms: int = 200
    whisper_model: str = DEFAULT_WHISPER_MODEL
    torch_threads: int = 4  # PyTorch thread cap in the STT process; leaves CPU for VAD/audio (0 = default)
    stt_timeout_s: float = 60.0  # per-segment hard limit for a stuck STT process (then: failed)
    prewarm_max_wait_s: float = 600.0  # how long a replacement worker defers loading behind active jobs
    # openai stack (optional)
    reasoner_model: str = DEFAULT_MODEL
    stt_model: str = "whisper-1"  # same as the official cascaded template
    tts_model: str = "tts-1"
    tts_voice: str = "nova"
    # shared
    seed: int = 7
    stability_s: float = 0.6
    backchannel: bool = True
    journal_dir: str = str(REPO_ROOT / "results" / "journals")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentSettings":
        env = os.environ if env is None else env
        d = cls()
        stack = env.get("FDAGENT_STACK", d.stack)
        if stack not in STACKS:
            raise ValueError(f"FDAGENT_STACK must be one of {STACKS}, got {stack!r}")
        return cls(
            stack=stack,
            latency_profile=env.get("FDB_LATENCY_PROFILE", d.latency_profile),
            gemini_planner_model=env.get("FDAGENT_GEMINI_PLANNER_MODEL", d.gemini_planner_model),
            gemini_reasoning_effort=env.get("FDAGENT_GEMINI_REASONING_EFFORT", d.gemini_reasoning_effort),
            gemini_tts_model=env.get("FDAGENT_GEMINI_TTS_MODEL", d.gemini_tts_model),
            gemini_voice=env.get("FDAGENT_GEMINI_VOICE", d.gemini_voice),
            tts_backend=_choice(env.get("FDAGENT_TTS_BACKEND", d.tts_backend), TTS_BACKENDS, "FDAGENT_TTS_BACKEND"),
            piper_voice=env.get("FDAGENT_PIPER_VOICE", d.piper_voice),
            tts_streaming=env.get("FDAGENT_TTS_STREAMING", "1") not in ("0", "false", "False"),
            tts_prebuffer_ms=int(env.get("FDAGENT_TTS_PREBUFFER_MS", d.tts_prebuffer_ms)),
            whisper_model=env.get("FDAGENT_WHISPER_MODEL", d.whisper_model),
            torch_threads=int(env.get("FDAGENT_TORCH_THREADS", d.torch_threads)),
            stt_timeout_s=float(env.get("FDAGENT_STT_TIMEOUT_S", d.stt_timeout_s)),
            prewarm_max_wait_s=float(env.get("FDAGENT_PREWARM_MAX_WAIT_S", d.prewarm_max_wait_s)),
            reasoner_model=env.get("FDAGENT_REASONER_MODEL", d.reasoner_model),
            stt_model=env.get("FDAGENT_STT_MODEL", d.stt_model),
            tts_model=env.get("FDAGENT_TTS_MODEL", d.tts_model),
            tts_voice=env.get("FDAGENT_TTS_VOICE", d.tts_voice),
            seed=int(env.get("FDAGENT_SEED", d.seed)),
            stability_s=float(env.get("FDAGENT_STABILITY_S", d.stability_s)),
            backchannel=env.get("FDAGENT_BACKCHANNEL", "1") not in ("0", "false", "False"),
            journal_dir=env.get("FDAGENT_JOURNAL_DIR", d.journal_dir),
        )

    def kernel_config(self) -> KernelConfig:
        return KernelConfig(stability_s=self.stability_s, backchannel=self.backchannel)


def required_env(settings: AgentSettings) -> tuple[str, ...]:
    keys = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    return keys + (("GOOGLE_API_KEY",) if settings.stack == "gemini_local" else ("OPENAI_API_KEY",))


class KernelDrivenAgent(Agent):
    """LiveKit agent shell: never generates its own replies."""

    def __init__(self) -> None:
        super().__init__(instructions="Replies are produced by the fdagent session kernel.")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:  # noqa: ANN001
        raise StopResponse()


def _vad():
    from livekit.plugins import silero

    # VAD parameters as in the official cascaded template (tuned for pre-recorded audio).
    return silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.55)


def _transcriber(model_id: str):
    """Whisper runs in a dedicated child process (its own GIL), never in the agent process."""
    from fdagent.voice.stt_process import SttProcessClient

    s = AgentSettings.from_env()
    return SttProcessClient(model_id, threads=s.torch_threads, timeout_s=s.stt_timeout_s)


# ── process prewarm (LiveKit setup_fnc) ─────────────────────────────────────
PREWARM_KEY = "fdagent_prewarmed"


@dataclass(frozen=True)
class Prewarmed:
    """Expensive, reusable resources loaded once per worker process, before any job.

    Sharing: on Linux each job runs in its own pre-started process, so nothing is shared
    between rooms. With the thread executor (Windows) jobs share the process: the Silero
    VAD is designed for this (each session opens its own stream) and the Whisper
    transcriber is a locked process-wide cache. Per-room objects (LocalWhisperSTT wrapper,
    AgentSession, planner, TTS client) are still built in the entrypoint; they are cheap.
    """

    stack: str
    whisper_model: str | None
    vad: Any
    whisper: Any  # WhisperTranscriber with weights loaded, or None for the openai stack
    load_s: float
    tts_backend: str | None = None  # "piper" when a local voice was preloaded
    piper: Any = None  # PiperTTS with its voice loaded (stateless per call; safe to share)


def limit_torch_threads(n: int) -> None:
    """Cap PyTorch intra-op threads in this process so Whisper inference/loading cannot starve
    voice detection and the audio loop of CPU. No-op if torch is unavailable."""
    if n <= 0:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(n)


def _piper(voice_path: str):
    from fdagent.providers.local_piper_tts import PiperTTS

    tts = PiperTTS(model_path=voice_path or None)
    tts.load()
    return tts


def load_prewarmed(settings: AgentSettings, vad_loader=None, transcriber_loader=None, piper_loader=None) -> Prewarmed:
    t0 = time.monotonic()
    vad = (vad_loader or _vad)()
    whisper = piper = None
    if settings.stack == "gemini_local":
        whisper = (transcriber_loader or _transcriber)(settings.whisper_model)
        whisper.load()  # starts the STT child process and waits until it has loaded Whisper
        import openai  # noqa: F401  (SDK used by the Gemini planner; import cost paid here, not per room)
        if settings.tts_backend == "piper":
            piper = (piper_loader or _piper)(settings.piper_voice)  # fails loudly if the voice is missing
    return Prewarmed(settings.stack, settings.whisper_model if whisper else None, vad, whisper,
                     round(time.monotonic() - t0, 2), "piper" if piper else None, piper)


def prewarm_process(proc: agents.JobProcess, registry: ActiveJobRegistry | None = None) -> None:
    """LiveKit ``setup_fnc``: runs once in each idle worker process before it takes a job.

    A replacement worker spawned while a job is running waits until no job is active before
    loading models, so its CPU-heavy setup never overlaps a live conversation."""
    settings = AgentSettings.from_env()
    waited = (registry or ActiveJobRegistry()).wait_until_idle(max_wait_s=settings.prewarm_max_wait_s)
    pw = load_prewarmed(settings)
    proc.userdata[PREWARM_KEY] = pw
    log.info("fdagent prewarm complete pid=%s stack=%s whisper=%s load_s=%s waited_for_jobs_s=%s", os.getpid(),
             pw.stack, pw.whisper_model, pw.load_s, waited)


async def resolve_prewarmed(userdata: Mapping[str, Any], settings: AgentSettings) -> Prewarmed:
    """Use the process's preloaded resources; fall back to a (logged) cold load if absent."""
    pw = userdata.get(PREWARM_KEY)
    local = settings.stack == "gemini_local"
    wanted = settings.whisper_model if local else None
    wanted_tts = "piper" if local and settings.tts_backend == "piper" else None
    if (isinstance(pw, Prewarmed) and pw.stack == settings.stack and pw.whisper_model == wanted
            and pw.tts_backend == wanted_tts):
        return pw
    log.warning("fdagent: no matching prewarmed resources in this process; cold-loading in the entrypoint")
    return await asyncio.to_thread(load_prewarmed, settings)


def build_stack(settings: AgentSettings, journal: Journal, prewarmed: Prewarmed,
                session_factory=None, stt_observer=None) -> tuple[Any, Any, Any]:
    """Return (reasoner, session, speech_sink) for the configured provider stack.

    Never loads models: VAD and Whisper come from ``prewarmed``."""
    session_factory = session_factory or AgentSession
    warn = lambda w: journal.annotate(name="reasoner_warning", warning=w)  # noqa: E731
    session_kw = dict(allow_interruptions=True, min_endpointing_delay=0.5, max_endpointing_delay=5.0)

    if settings.stack == "gemini_local":
        from fdagent.providers.gemini_reasoner import GeminiReasoner
        from fdagent.providers.gemini_tts import GeminiTTS
        from fdagent.voice.local_whisper import LocalWhisperSTT
        from fdagent.voice.speech_sink import PrerenderedSpeechSink

        def attempt(a: dict[str, Any]) -> None:  # per-HTTP-attempt planner status/retries
            journal.annotate(name="planner_attempt", t=time.monotonic(), wall=time.time(), **a)

        reasoner = GeminiReasoner(FDB_TOOL_SPECS, model=settings.gemini_planner_model,
                                  reasoning_effort=settings.gemini_reasoning_effort, on_warning=warn,
                                  on_attempt=attempt)
        # Lightweight per-room wrapper around the shared STT process; reports segment lifecycle.
        whisper = LocalWhisperSTT(transcriber=prewarmed.whisper, observer=stt_observer)
        # No TTS plugin: speech is synthesized by the selected backend (local Piper by default,
        # Gemini optionally), streamed, and played via session.say(audio=...).
        session = session_factory(vad=prewarmed.vad, stt=whisper, **session_kw)
        if settings.tts_backend == "piper":
            tts = prewarmed.piper
        else:
            tts = GeminiTTS(model=settings.gemini_tts_model, voice=settings.gemini_voice)
        sink = PrerenderedSpeechSink(session, tts, streaming=settings.tts_streaming,
                                     prebuffer_ms=settings.tts_prebuffer_ms)
        return reasoner, session, sink

    from livekit.plugins import openai

    from fdagent.providers.openai_reasoner import OpenAIReasoner
    from fdagent.voice.speech_sink import LiveKitSpeechSink

    reasoner = OpenAIReasoner(FDB_TOOL_SPECS, model=settings.reasoner_model, seed=settings.seed, on_warning=warn)
    session = session_factory(vad=prewarmed.vad, stt=openai.STT(model=settings.stt_model, language="en"),
                              tts=openai.TTS(model=settings.tts_model, voice=settings.tts_voice), **session_kw)
    return reasoner, session, LiveKitSpeechSink(session)


def server_options(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Worker options. Each idle process holds its own Whisper copy, so keep the pool small
    (LiveKit's production default is 18). setup_fnc may first wait for active jobs to finish
    (see prewarm_process) and then load models, so the initialize timeout must cover both
    (LiveKit's 10 s default is shorter than a cold Whisper load alone)."""
    env = os.environ if env is None else env
    return dict(
        setup_fnc=prewarm_process,
        num_idle_processes=int(env.get("FDAGENT_IDLE_PROCESSES", "2")),
        initialize_process_timeout=float(env.get("FDAGENT_PROCESS_INIT_TIMEOUT_S", "900")),
    )


server = AgentServer(**server_options())


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    settings = AgentSettings.from_env()
    missing = [k for k in required_env(settings) if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"missing environment variables for stack {settings.stack!r}: {missing}")
    room = ctx.room.name
    log.info("fdagent job start room=%s pid=%s", room, os.getpid())
    # Mark this job active so replacement workers defer their model loading until it ends.
    registry = ActiveJobRegistry()
    job_key = registry.register(f"{os.getpid()}-{room}")

    async def _unregister() -> None:
        registry.unregister(job_key)

    ctx.add_shutdown_callback(_unregister)
    FdbMockBackend.seed(settings.seed)

    journal = Journal(Path(settings.journal_dir) / f"{room}.jsonl")
    kernel = SessionKernel(room, list(FDB_TOOL_SPECS), settings.kernel_config(), journal=journal,
                           settings=asdict(settings))
    prewarmed = await resolve_prewarmed(ctx.proc.userdata, settings)
    local_stt = settings.stack == "gemini_local"
    # The STT adapter reports started/final/failed per segment; forwarded to the bridge (created below).
    reasoner, session, sink = build_stack(settings, journal, prewarmed,
                                          stt_observer=lambda kind, **f: bridge.on_stt_event(kind, **f))
    if local_stt and os.name != "nt" and hasattr(prewarmed.whisper, "close"):
        # One job per worker process on POSIX: stop the STT child when this job ends. (Windows runs
        # jobs as threads of one process; there the child is closed at process exit.)
        async def _close_stt() -> None:
            await asyncio.to_thread(prewarmed.whisper.close)

        ctx.add_shutdown_callback(_close_stt)

    runtime = SessionRuntime(kernel, reasoner, FdbMockBackend(room, latency_profile=settings.latency_profile), sink)
    heartbeat = FdbLatencyLog(room)
    runtime.add_mark_listener(heartbeat.on_mark)
    # Local STT supplies transcripts itself (with lifecycle); LiveKit's copies would double-count.
    bridge = IngressBridge(runtime.post, on_final=heartbeat.on_user_final, livekit_transcripts=not local_stt)
    session.on("user_state_changed", lambda ev: bridge.on_user_state(ev.new_state))
    session.on("user_input_transcribed", lambda ev: bridge.on_transcript(ev.transcript, ev.is_final))
    # Observability only: when LiveKit actually starts/stops playing audio, and TTS first-audio marks.
    session.on("agent_state_changed", lambda ev: runtime.mark(name="agent_state", state=ev.new_state))
    if hasattr(sink, "observer"):
        sink.observer = runtime.mark

    await runtime.start()
    ctx.add_shutdown_callback(runtime.aclose)
    log.info("fdagent joining room %s (%s)", room, asdict(settings))
    await session.start(room=ctx.room, agent=KernelDrivenAgent())
    runtime.mark(name="session_started", room=room)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env.local")
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()
