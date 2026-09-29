"""Gemini planner adapter: request shape, parsing/coercion, fallbacks, and proof that the
kernel remains the sole tool dispatcher. No network: a fake OpenAI-compatible client."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from fdagent.core.actions import RequestReasoning
from fdagent.core.config import KernelConfig
from fdagent.core.events import UserSpeechStarted, UserTranscript, UserTurnEnded
from fdagent.core.journal import Journal
from fdagent.core.kernel import SessionKernel
from fdagent.providers.gemini_reasoner import (
    DEFAULT_GEMINI_PLANNER_MODEL,
    GEMINI_OPENAI_BASE_URL,
    GeminiReasoner,
    gemini_api_key,
)
from fdagent.providers.openai_reasoner import PlanParseError
from fdagent.runtime.loop import SessionRuntime

from .fakes import RecordingSink, decisions, until
from .test_phase3_integration import RoomBackend, town_planner
from .test_reasoner_parsing import TOOLS_X


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeGemini:
    """Stands in for AsyncOpenAI(base_url=Gemini). ``script`` maps payload → content str,
    or returns an Exception to raise, or a dict with 'tool_calls' to emulate a function call."""

    def __init__(self, script: Callable[[dict, dict], Any]) -> None:
        self.script = script
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.requests.append(kw)
        payload = json.loads(kw["messages"][-1]["content"])
        out = self.script(payload, kw)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, dict) and "tool_calls" in out:
            msg = SimpleNamespace(content=None, tool_calls=out["tool_calls"])
        else:
            msg = SimpleNamespace(content=out if isinstance(out, str) else json.dumps(out), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def req(conversation=(), calls=()):
    snap = {"generation": 1, "conversation": list(conversation), "calls": list(calls), "orphan_effects": [],
            "transcript": [c["text"] for c in conversation]}
    return RequestReasoning("req-1", 1, snap)


def run(coro):
    return asyncio.run(coro)


def test_request_shape_and_typed_parsing():
    plan = {"keep": [], "new_calls": [{"tool": "find_rooms", "args": {"town": " Eastvale ", "beds": "3", "budget": "$1,100"}}],
            "reply": None}
    client = FakeGemini(lambda p, kw: plan)
    draft = run(GeminiReasoner(TOOLS_X, client=client).propose(req([{"role": "user", "text": "um rooms"}])))
    [c] = draft.calls
    assert (c.tool, c.args) == ("find_rooms", {"town": "Eastvale", "beds": 3, "budget": 1100.0})
    kw = client.requests[0]
    assert kw["model"] == DEFAULT_GEMINI_PLANNER_MODEL == "gemini-3.5-flash-lite"
    assert kw["temperature"] == 0 and kw["reasoning_effort"] == "low"
    assert kw["response_format"] == {"type": "json_object"}
    # The planner is never offered tools and never sends an unsupported seed.
    assert not {"tools", "tool_choice", "functions", "seed"} & set(kw)


def test_json_mode_rejected_falls_back_once_to_prompt_only_json():
    def script(p, kw):
        if "response_format" in kw:
            return StatusError(400)
        return '```json\n{"keep": [], "new_calls": [], "reply": "Hi!"}\n```'

    warnings: list[str] = []
    client = FakeGemini(script)
    r = GeminiReasoner(TOOLS_X, client=client, on_warning=warnings.append)
    assert run(r.propose(req())).reply == "Hi!"
    assert run(r.propose(req())).reply == "Hi!"  # remembered: no second rejected request
    assert ["response_format" in k for k in client.requests] == [True, False, False]
    assert any("json_object" in w for w in warnings)


def test_non_400_errors_propagate():
    client = FakeGemini(lambda p, kw: StatusError(429))
    with pytest.raises(StatusError):
        run(GeminiReasoner(TOOLS_X, client=client).propose(req()))
    assert len(client.requests) == 1  # SDK-level retries happen inside the real client, not here


def test_native_tool_call_attempt_is_rejected():
    client = FakeGemini(lambda p, kw: {"tool_calls": [{"function": {"name": "find_rooms", "arguments": "{}"}}]})
    with pytest.raises(PlanParseError, match="only the kernel may dispatch"):
        run(GeminiReasoner(TOOLS_X, client=client).propose(req()))


def test_missing_key_fails_explicitly(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        run(GeminiReasoner(TOOLS_X).propose(req()))


def test_key_lookup_and_real_client_configuration():
    assert gemini_api_key({"GOOGLE_API_KEY": "a", "GEMINI_API_KEY": "b"}) == "a"
    assert gemini_api_key({"GEMINI_API_KEY": "b"}) == "b"
    assert gemini_api_key({}) is None
    client = GeminiReasoner(TOOLS_X, api_key="test-placeholder")._get_client()  # constructs, no network
    assert str(client.base_url) == GEMINI_OPENAI_BASE_URL and client.max_retries == 2


def test_agent_settings_default_to_zero_cost_stack():
    pytest.importorskip("livekit.agents")
    from fdagent.voice.livekit_agent import AgentSettings, required_env

    s = AgentSettings.from_env({})
    assert (s.stack, s.gemini_planner_model, s.gemini_tts_model, s.whisper_model) == (
        "gemini_local", "gemini-3.5-flash-lite", "gemini-3.8-flash-lite-tts", "openai/whisper-tiny.en")
    # The previous default stays selectable through the environment override.
    assert AgentSettings.from_env({"FDAGENT_GEMINI_PLANNER_MODEL": "gemini-3.8-flash"}).gemini_planner_model == "gemini-3.8-flash"
    assert "OPENAI_API_KEY" not in required_env(s) and "GOOGLE_API_KEY" in required_env(s)
    assert "OPENAI_API_KEY" in required_env(AgentSettings.from_env({"FDAGENT_STACK": "openai"}))
    with pytest.raises(ValueError):
        AgentSettings.from_env({"FDAGENT_STACK": "mystery"})


def test_planner_model_override_reaches_the_request():
    client = FakeGemini(lambda p, kw: {"keep": [], "new_calls": [], "reply": "Hi"})
    run(GeminiReasoner(TOOLS_X, client=client, model="gemini-3.8-flash").propose(req()))
    assert client.requests[0]["model"] == "gemini-3.8-flash"


# ── the kernel is the only dispatcher ──────────────────────────────────────
def _runtime(script, backend):
    kernel = SessionKernel("g-sole", list(TOOLS_X), KernelConfig(stability_s=0.05), journal=Journal())
    reasoner = GeminiReasoner(TOOLS_X, client=FakeGemini(script))
    return SessionRuntime(kernel, reasoner, backend, RecordingSink()), reasoner


class IdemRecordingBackend(RoomBackend):
    def __init__(self):
        super().__init__()
        self.idem_keys: list[str] = []

    async def acall(self, tool, args, idem):
        self.idem_keys.append(idem)
        return await super().acall(tool, args, idem)


def test_kernel_is_the_sole_tool_dispatcher():
    """Every backend invocation corresponds 1:1 to a kernel dispatch decision; the planner holds
    no reference to any tool backend or executor, and a native tool-call attempt executes nothing."""

    async def main(script):
        backend = IdemRecordingBackend()
        rt, reasoner = _runtime(script, backend)
        await rt.start()
        rt.post(UserSpeechStarted())
        rt.post(UserTranscript("rooms in Eastvale, no, Westbrook"))
        rt.post(UserTurnEnded())
        await until(lambda: rt.kernel.state.replied_generation >= 0 or decisions(rt, "reasoner_failed"))
        await rt.idle()
        await rt.aclose()
        return rt, reasoner, backend

    rt, reasoner, backend = run(main(lambda p, kw: town_planner(p)))
    dispatched = [d["call_id"] for d in decisions(rt, "dispatch")]
    assert dispatched and backend.idem_keys == dispatched  # same calls, same order, nothing extra
    assert all(not isinstance(v, (IdemRecordingBackend, SessionRuntime)) for v in vars(reasoner).values())

    rt2, _, backend2 = run(main(lambda p, kw: {"tool_calls": [{"function": {"name": "find_rooms"}}]}))
    assert backend2.calls == [] and decisions(rt2, "dispatch") == []
    assert "only the kernel may dispatch" in decisions(rt2, "reasoner_failed")[0]["error"]
