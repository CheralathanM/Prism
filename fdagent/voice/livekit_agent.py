#!/usr/bin/env python3
"""FDB-v3 LiveKit agent: LiveKit handles audio I/O; the session kernel handles everything else.

Follows the official template pattern (``v3/cascaded_agent.py``): an ``AgentServer`` with an
``rtc_session`` entrypoint, one session per room, and tool telemetry in the harness format.
Differences:
  * AgentSession runs VAD + STT + TTS only (no LLM): LiveKit's own reply loop is disabled.
  * VAD/STT callbacks become kernel events (ingress); kernel Speak actions become
    ``session.say`` (egress). Tool calls go kernel → executor → official mock backend.
  * The OpenAI planner only proposes plans; it cannot execute tools.

Run (from the repo root, with .env.local configured):
    python -m fdagent.voice.livekit_agent start      # or: dev / console
Provider name for the harness: ``--provider fdagent``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, StopResponse

from fdagent.adapters.fdb_tools import FDB_TOOL_SPECS, FdbMockBackend
from fdagent.core.config import KernelConfig
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.providers.openai_reasoner import DEFAULT_MODEL, OpenAIReasoner
from fdagent.runtime.loop import SessionRuntime
from fdagent.voice.fdb_heartbeat import FdbLatencyLog
from fdagent.voice.ingress import IngressBridge
from fdagent.voice.speech_sink import LiveKitSpeechSink

log = logging.getLogger("fdagent.livekit")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AgentSettings:
    provider: str = "fdagent"
    latency_profile: str = "instant"
    reasoner_model: str = DEFAULT_MODEL
    stt_model: str = "whisper-1"  # same as the official cascaded template
    tts_model: str = "tts-1"
    tts_voice: str = "nova"
    seed: int = 7
    stability_s: float = 0.6
    backchannel: bool = True
    journal_dir: str = str(REPO_ROOT / "results" / "journals")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentSettings":
        env = os.environ if env is None else env
        d = cls()
        return cls(
            latency_profile=env.get("FDB_LATENCY_PROFILE", d.latency_profile),
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


class KernelDrivenAgent(Agent):
    """LiveKit agent shell: never generates its own replies."""

    def __init__(self) -> None:
        super().__init__(instructions="Replies are produced by the fdagent session kernel.")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:  # noqa: ANN001
        raise StopResponse()


def build_voice(settings: AgentSettings):
    from livekit.plugins import openai, silero

    # VAD parameters as in the official cascaded template (tuned for pre-recorded audio).
    vad = silero.VAD.load(min_speech_duration=0.05, min_silence_duration=0.55)
    stt = openai.STT(model=settings.stt_model, language="en")
    tts = openai.TTS(model=settings.tts_model, voice=settings.tts_voice)
    return vad, stt, tts


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    settings = AgentSettings.from_env()
    room = ctx.room.name
    FdbMockBackend.seed(settings.seed)

    journal = Journal(Path(settings.journal_dir) / f"{room}.jsonl")
    kernel = SessionKernel(room, list(FDB_TOOL_SPECS), settings.kernel_config(), journal=journal,
                           settings=asdict(settings))
    reasoner = OpenAIReasoner(
        FDB_TOOL_SPECS, model=settings.reasoner_model, seed=settings.seed,
        on_warning=lambda w: journal.annotate(name="reasoner_warning", warning=w),
    )
    vad, stt, tts = build_voice(settings)
    session = AgentSession(vad=vad, stt=stt, tts=tts, allow_interruptions=True,
                           min_endpointing_delay=0.5, max_endpointing_delay=5.0)

    runtime = SessionRuntime(kernel, reasoner, FdbMockBackend(room, latency_profile=settings.latency_profile),
                             LiveKitSpeechSink(session))
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
