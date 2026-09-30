"""Gemini planner via Google's OpenAI-compatible endpoint (free tier, no billing account).

Same contract as ``OpenAIReasoner``: the shared planner prompt and parser
(``build_messages`` / ``parse_plan``) produce the identical Draft / JSON-plan interface the
kernel consumes. The model is never given tool definitions, so it cannot issue native
function calls; if it tries anyway, the output is rejected and the kernel reports an honest
fallback. Only the kernel dispatches tools.

Config: ``GOOGLE_API_KEY`` (``GEMINI_API_KEY`` also accepted).
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from typing import Any, Callable, Iterable, Mapping

from fdagent.core.actions import RequestReasoning
from fdagent.core.model import ToolSpec
from fdagent.runtime.reasoner import Draft

from .openai_reasoner import PlanParseError, build_messages, parse_plan

log = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Free tier per ai.google.dev pricing. flash-lite chosen for latency (measured ~1.1 s vs
# 16.8 s + 503 "high demand" for gemini-3.8-flash on the free tier); gemini-3.8-flash
# remains supported via FDAGENT_GEMINI_PLANNER_MODEL.
DEFAULT_GEMINI_PLANNER_MODEL = "gemini-3.5-flash-lite"
DEFAULT_REASONING_EFFORT = "low"  # Gemini 3 models cannot disable thinking; keep it small


_CURRENT_REQUEST: contextvars.ContextVar[str | None] = contextvars.ContextVar("fdagent_planner_request", default=None)


def attempt_event_hooks(on_attempt: Callable[[dict[str, Any]], None]) -> dict[str, list]:
    """httpx event hooks reporting every HTTP attempt the SDK makes (including its retries).

    Reported fields: planner request_id, SDK retry count (``x-stainless-retry-count``), HTTP
    status, ``retry-after``, and attempt duration. Never includes headers carrying credentials.
    """

    async def on_request(request):
        request.extensions["fdagent_t0"] = time.monotonic()

    async def on_response(response):
        t0 = response.request.extensions.get("fdagent_t0")
        on_attempt({
            "request_id": _CURRENT_REQUEST.get(),
            "retry_count": response.request.headers.get("x-stainless-retry-count"),
            "status": response.status_code,
            "retry_after": response.headers.get("retry-after"),
            "attempt_s": round(time.monotonic() - t0, 3) if t0 is not None else None,
        })

    return {"request": [on_request], "response": [on_response]}


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
        on_attempt: Callable[[dict[str, Any]], None] | None = None,
        transport: Any = None,
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
        self._on_attempt = on_attempt  # per-HTTP-attempt status/retry observability
        self._transport = transport  # tests only (httpx.MockTransport)

    def _get_client(self) -> Any:
        if self._client is None:
            key = self._api_key or gemini_api_key()
            if not key:
                raise RuntimeError("GOOGLE_API_KEY is not set (needed for the Gemini planner)")
            from openai import AsyncOpenAI  # only the SDK; requests go to Google, not OpenAI

            kw: dict[str, Any] = {}
            if self._on_attempt is not None or self._transport is not None:
                import httpx

                kw["http_client"] = httpx.AsyncClient(
                    transport=self._transport,
                    event_hooks=attempt_event_hooks(self._on_attempt) if self._on_attempt else None,
                )
            # max_retries: the SDK retries 429 / 5xx with backoff (free-tier rate limits).
            self._client = AsyncOpenAI(api_key=key, base_url=GEMINI_OPENAI_BASE_URL,
                                       max_retries=self._max_retries, **kw)
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
        _CURRENT_REQUEST.set(request.request_id)  # task-local: labels attempts in the hooks
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
