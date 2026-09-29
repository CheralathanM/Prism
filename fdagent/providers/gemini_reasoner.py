"""Gemini planner via Google's OpenAI-compatible endpoint (free tier, no billing account).

Same contract as ``OpenAIReasoner``: the shared planner prompt and parser
(``build_messages`` / ``parse_plan``) produce the identical Draft / JSON-plan interface the
kernel consumes. The model is never given tool definitions, so it cannot issue native
function calls; if it tries anyway, the output is rejected and the kernel reports an honest
fallback. Only the kernel dispatches tools.

Config: ``GOOGLE_API_KEY`` (``GEMINI_API_KEY`` also accepted).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, Mapping

from fdagent.core.actions import RequestReasoning
from fdagent.core.model import ToolSpec
from fdagent.runtime.reasoner import Draft

from .openai_reasoner import PlanParseError, build_messages, parse_plan

log = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-3.8-flash"  # stable; free tier per ai.google.dev pricing
DEFAULT_REASONING_EFFORT = "low"  # Gemini 3 models cannot disable thinking; keep it small


def gemini_api_key(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    return env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY") or None


class GeminiReasoner:
    def __init__(
        self,
        tools: Iterable[ToolSpec],
        client: Any = None,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        timeout_s: float = 20.0,
        json_mode: bool = True,
        max_retries: int = 2,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.tools = tuple(tools)
        self.specs = {t.name: t for t in self.tools}
        self.model = model or DEFAULT_GEMINI_PLANNER_MODEL
        self.reasoning_effort = reasoning_effort or None
        self.timeout_s = timeout_s
        # The compatibility docs do not confirm response_format=json_object; we try it and
        # fall back once to prompt-enforced JSON (parse_plan strips fences) if rejected.
        self.json_mode = json_mode
        self._max_retries = max_retries
        self._api_key = api_key
        self._client = client
        self._on_warning = on_warning or (lambda w: log.warning("gemini reasoner: %s", w))

    def _get_client(self) -> Any:
        if self._client is None:
            key = self._api_key or gemini_api_key()
            if not key:
                raise RuntimeError("GOOGLE_API_KEY is not set (needed for the Gemini planner)")
            from openai import AsyncOpenAI  # only the SDK; requests go to Google, not OpenAI

            # max_retries: the SDK retries 429 / 5xx with backoff (free-tier rate limits).
            self._client = AsyncOpenAI(api_key=key, base_url=GEMINI_OPENAI_BASE_URL, max_retries=self._max_retries)
        return self._client

    def _kwargs(self, request: RequestReasoning) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            model=self.model,
            messages=build_messages(request.snapshot, self.tools),
            temperature=0,
            timeout=self.timeout_s,
        )
        # Deliberately no "tools" / "tool_choice": the planner may only return a JSON plan.
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    async def propose(self, request: RequestReasoning) -> Draft:
        client = self._get_client()
        try:
            resp = await client.chat.completions.create(**self._kwargs(request))
        except Exception as e:
            if not (self.json_mode and getattr(e, "status_code", None) == 400):
                raise
            self.json_mode = False
            self._on_warning(f"response_format=json_object rejected ({e.__class__.__name__}); using prompt-only JSON")
            resp = await client.chat.completions.create(**self._kwargs(request))

        message = resp.choices[0].message
        if getattr(message, "tool_calls", None):
            raise PlanParseError("model attempted a direct tool call; only the kernel may dispatch tools")
        draft, warnings = parse_plan(message.content, request.snapshot, self.specs)
        for w in warnings:
            self._on_warning(w)
        return draft
