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

log = logging.getLogger("fdagent.livekit")
REPO_ROOT = Path(__file__).resolve().parents[2]
STACKS = ("gemini_local", "openai")
DEFAULT_WHISPER_MODEL = "openai/whisper-tiny.en"  # mirrors voice.local_whisper (avoids importing it eagerly)


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
    whisper_model: str = DEFAULT_WHISPER_MODEL
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
            whisper_model=env.get("FDAGENT_WHISPER_MODEL", d.whisper_model),
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


async def build_stack(settings: AgentSettings, journal: Journal) -> tuple[Any, AgentSession, Any]:
    """Return (reasoner, session, speech_sink) for the configured provider stack."""
    warn = lambda w: journal.annotate(name="reasoner_warning", warning=w)  # noqa: E731
    session_kw = dict(allow_interruptions=True, min_endpointing_delay=0.5, max_endpointing_delay=5.0)

    if settings.stack == "gemini_local":
        from fdagent.providers.gemini_reasoner import GeminiReasoner
        from fdagent.providers.gemini_tts import GeminiTTS
        from fdagent.voice.local_whisper import LocalWhisperSTT
        from fdagent.voice.speech_sink import PrerenderedSpeechSink

        reasoner = GeminiReasoner(FDB_TOOL_SPECS, model=settings.gemini_planner_model,
                                  reasoning_effort=settings.gemini_reasoning_effort, on_warning=warn)
        whisper = LocalWhisperSTT(model_id=settings.whisper_model)
        await asyncio.to_thread(whisper.prewarm)  # load weights before the first utterance
        # No TTS plugin: speech is pre-rendered by Gemini TTS and played via session.say(audio=...).
        session = AgentSession(vad=_vad(), stt=whisper, **session_kw)
        sink = PrerenderedSpeechSink(session, GeminiTTS(model=settings.gemini_tts_model, voice=settings.gemini_voice))
        return reasoner, session, sink

    from livekit.plugins import openai

    from fdagent.providers.openai_reasoner import OpenAIReasoner
    from fdagent.voice.speech_sink import LiveKitSpeechSink

    reasoner = OpenAIReasoner(FDB_TOOL_SPECS, model=settings.reasoner_model, seed=settings.seed, on_warning=warn)
    session = AgentSession(vad=_vad(), stt=openai.STT(model=settings.stt_model, language="en"),
                           tts=openai.TTS(model=settings.tts_model, voice=settings.tts_voice), **session_kw)
    return reasoner, session, LiveKitSpeechSink(session)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    settings = AgentSettings.from_env()
    missing = [k for k in required_env(settings) if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"missing environment variables for stack {settings.stack!r}: {missing}")
    room = ctx.room.name
    FdbMockBackend.seed(settings.seed)

    journal = Journal(Path(settings.journal_dir) / f"{room}.jsonl")
    kernel = SessionKernel(room, list(FDB_TOOL_SPECS), settings.kernel_config(), journal=journal,
                           settings=asdict(settings))
    reasoner, session, sink = await build_stack(settings, journal)

    runtime = SessionRuntime(kernel, reasoner, FdbMockBackend(room, latency_profile=settings.latency_profile), sink)
    heartbeat = FdbLatencyLog(room)
    runtime.add_mark_listener(heartbeat.on_mark)
    bridge = IngressBridge(runtime.post, on_final=heartbeat.on_user_final)
    session.on("user_state_changed", lambda ev: bridge.on_user_state(ev.new_state))
    session.on("user_input_transcribed", lambda ev: bridge.on_transcript(ev.transcript, ev.is_final))

    await runtime.start()
    ctx.add_shutdown_callback(runtime.aclose)
    log.info("fdagent joining room %s (%s)", room, asdict(settings))
    await session.start(room=ctx.room, agent=KernelDrivenAgent())


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env.local")
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()
