"""Cold-start regression: expensive models load in LiveKit's setup_fnc, never in the room entrypoint.

Observed failure this guards against: Whisper + Silero loading inside the entrypoint delayed
RoomIO by ~12 s, so the agent missed a request that ended 6.5 s into the audio.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("livekit.agents")

import fdagent.voice.livekit_agent as la  # noqa: E402
import fdagent.voice.local_whisper as lw  # noqa: E402
from fdagent.adapters.fdb_tools import DEFAULT_FDB_V3_DIR  # noqa: E402
from fdagent.core.journal import Journal  # noqa: E402


class FakeTranscriber:
    def __init__(self, model_id):
        self.model_id, self.loads = model_id, 0

    def load(self):
        self.loads += 1


def _forbid_cold_loading(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("cold model loading on the room/job path")

    monkeypatch.setattr(la, "_vad", boom)
    monkeypatch.setattr(la, "_transcriber", boom)
    monkeypatch.setattr(la, "_piper", boom)
    monkeypatch.setattr(la, "load_prewarmed", boom)
    monkeypatch.setattr(lw, "get_transcriber", boom)
    monkeypatch.setattr(lw, "_transformers_pipeline", boom)


class FakePiper:
    stream_sample_rate = 22050

    async def synthesize(self, text):  # pragma: no cover - not exercised here
        raise AssertionError


def _prewarmed(stack="gemini_local", model=la.DEFAULT_WHISPER_MODEL, tts_backend="piper"):
    local = stack == "gemini_local"
    return la.Prewarmed(stack, model if local else None, vad=object(),
                        whisper=FakeTranscriber(model) if local else None, load_s=0.0,
                        tts_backend=tts_backend if local else None,
                        piper=FakePiper() if local and tts_backend == "piper" else None)


def test_setup_fnc_loads_vad_and_whisper_once_into_userdata(monkeypatch, tmp_path):
    vads, transcribers = [], []
    monkeypatch.setattr(la, "_vad", lambda: vads.append(object()) or vads[-1])
    monkeypatch.setattr(la, "_transcriber", lambda m: transcribers.append(FakeTranscriber(m)) or transcribers[-1])
    monkeypatch.setattr(la, "limit_torch_threads", lambda n: None)
    pipers = []
    monkeypatch.setattr(la, "_piper", lambda path: pipers.append(FakePiper()) or pipers[-1])
    proc = SimpleNamespace(userdata={})
    la.prewarm_process(proc, registry=la.ActiveJobRegistry(tmp_path))
    pw = proc.userdata[la.PREWARM_KEY]
    assert pw.vad is vads[0] and len(vads) == 1
    assert pw.whisper is transcribers[0] and pw.whisper.loads == 1  # weights loaded here, before any job
    assert (pw.stack, pw.whisper_model) == ("gemini_local", la.DEFAULT_WHISPER_MODEL)
    assert pw.tts_backend == "piper" and pw.piper is pipers[0] and len(pipers) == 1  # local TTS voice preloaded


def test_server_is_configured_with_prewarm_small_pool_and_long_init_timeout():
    opts = la.server_options({})
    assert opts["setup_fnc"] is la.prewarm_process
    assert opts["num_idle_processes"] == 2 and opts["initialize_process_timeout"] == 900.0
    over = la.server_options({"FDAGENT_IDLE_PROCESSES": "1", "FDAGENT_PROCESS_INIT_TIMEOUT_S": "60"})
    assert (over["num_idle_processes"], over["initialize_process_timeout"]) == (1, 60.0)


def test_build_stack_reuses_prewarmed_resources_without_loading(monkeypatch):
    _forbid_cold_loading(monkeypatch)
    pw = _prewarmed()
    made = []
    session_factory = lambda **kw: made.append(kw) or SimpleNamespace(**kw)  # noqa: E731
    reasoner, session, sink = la.build_stack(la.AgentSettings.from_env({}), Journal(), pw, session_factory)
    [kw] = made
    assert kw["vad"] is pw.vad
    assert kw["stt"]._transcriber is pw.whisper  # per-room wrapper around the shared, loaded weights
    assert "tts" not in kw  # speech is synthesized by the TTS backend and played via session.say(audio=...)
    assert sink._tts is pw.piper  # default backend: the preloaded local Piper voice
    assert pw.whisper.loads == 0  # nothing (re)loaded on the room path


def test_resolve_prefers_prewarmed_and_logs_cold_fallback(monkeypatch, caplog):
    settings = la.AgentSettings.from_env({})
    pw = _prewarmed()
    got = asyncio.run(la.resolve_prewarmed({la.PREWARM_KEY: pw}, settings))
    assert got is pw

    sentinel = _prewarmed()
    monkeypatch.setattr(la, "load_prewarmed", lambda s: sentinel)
    with caplog.at_level("WARNING", logger="fdagent.livekit"):
        assert asyncio.run(la.resolve_prewarmed({}, settings)) is sentinel
        # a prewarm for a different Whisper model does not match either
        other = _prewarmed(model="openai/whisper-small.en")
        assert asyncio.run(la.resolve_prewarmed({la.PREWARM_KEY: other}, settings)) is sentinel
    assert sum("cold-loading" in r.message for r in caplog.records) == 2


@pytest.mark.skipif(not (DEFAULT_FDB_V3_DIR / "mock_apis.py").exists(), reason="FDB-v3 not fetched")
def test_room_entrypoint_performs_no_cold_model_loading(monkeypatch, tmp_path):
    """Drive the real entrypoint with a fake JobContext and a fake AgentSession; any attempt to
    load VAD/Whisper weights (or fall back to a cold load) raises."""
    _forbid_cold_loading(monkeypatch)
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GOOGLE_API_KEY"):
        monkeypatch.setenv(k, "test-placeholder")
    monkeypatch.delenv("FDAGENT_STACK", raising=False)
    monkeypatch.setenv("FDAGENT_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setenv("FDAGENT_RUNTIME_DIR", str(tmp_path / "rt"))

    started, shutdown = [], []

    class FakeSession:
        def __init__(self, **kw):
            self.kw, self.handlers = kw, {}

        def on(self, name, fn):
            self.handlers[name] = fn

        async def start(self, room, agent):
            started.append((self, room, agent))

    monkeypatch.setattr(la, "AgentSession", FakeSession)
    pw = _prewarmed()
    ctx = SimpleNamespace(room=SimpleNamespace(name="eval-prewarm1"), proc=SimpleNamespace(userdata={la.PREWARM_KEY: pw}),
                          add_shutdown_callback=shutdown.append)

    registry = la.ActiveJobRegistry(tmp_path / "rt" / "fdagent-active-jobs")
    active_during_job = []

    async def main():
        await la.entrypoint(ctx)
        active_during_job.extend(registry.active())
        for cb in shutdown:
            await cb()

    asyncio.run(main())
    assert len(active_during_job) == 1 and active_during_job[0].endswith("eval-prewarm1")
    assert registry.active() == []  # unregistered at shutdown
    [(session, room, agent)] = started
    assert session.kw["vad"] is pw.vad and session.kw["stt"]._transcriber is pw.whisper
    assert session.kw["stt"].observer is not None  # segment lifecycle reaches the kernel via the bridge
    assert set(session.handlers) == {"user_state_changed", "user_input_transcribed", "agent_state_changed"}
    assert room is ctx.room and isinstance(agent, la.KernelDrivenAgent)
    assert pw.whisper.loads == 0
    names = [r.get("name") for r in Journal.load(tmp_path / "eval-prewarm1.jsonl") if r["kind"] == "note"]
    assert "session_started" in names
