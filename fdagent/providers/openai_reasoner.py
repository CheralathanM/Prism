"""OpenAI planner: snapshot in, Draft (proposed plan + optional speech) out.

The model never executes tools and never sees call ids. It returns JSON:

    {"keep": [<indices of existing calls that are still exactly right>],
     "new_calls": [{"tool": "<name>", "args": {...}}],
     "reply": "<speech>" | null,
     "reply_kind": "final" | "progress" | "clarify"}

``keep`` + ``new_calls`` becomes the complete desired call set the kernel reconciles:
kept calls reproduce their exact identity (tool, args, occurrence) so the kernel reuses
the existing operation; omitted calls are superseded by the kernel. Chains are planned
iteratively: the kernel asks again once results are admitted, and the model then adds
the next step using concrete values from those results.

Arguments are coerced to the tool schema (spoken "2" → 2) but never guessed: values that
cannot be coerced are passed through unchanged so the kernel's schema gate rejects them
explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Iterable

from fdagent.adapters.tool_protocol import to_openai_tool
from fdagent.core.actions import RequestReasoning
from fdagent.core.events import ProposedCall
from fdagent.core.model import ToolSpec
from fdagent.runtime.reasoner import Draft

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"  # same model as the official cascaded FDB-v3 baseline

SYSTEM_PROMPT = """\
You are the planning module of a real-time voice assistant. You do not execute anything:
you return a JSON plan, and a separate controller decides when (and whether) each tool
call is executed, cancels calls you drop, and never executes the same call twice.

You receive:
- "conversation": what the user said (live speech-to-text, split into segments; it may
  contain fillers, pauses, hesitations, false starts and self-corrections) and what the
  assistant already said.
- "calls": tool calls already planned or executed for the current request, each with an
  index, arguments, status (planned / dispatched / succeeded / failed / unknown / invalid),
  and result or error.
- "tools": the available tools and their JSON schemas.

Return ONLY a JSON object:
{"keep": [indices], "new_calls": [{"tool": "...", "args": {...}}], "reply": string or null,
 "reply_kind": "final" | "progress" | "clarify"}

Rules:
1. Understand the user's CURRENT intent. When the user corrects themselves, the latest
   correction wins and the earlier value is discarded. Ignore fillers and abandoned false
   starts. Combine details spread across several segments.
2. "keep" + "new_calls" must be the COMPLETE set of calls the current intent needs right
   now. Keep existing calls that are still exactly right (never re-add them as new calls).
   Leave out calls that a correction made wrong; the controller cancels them.
3. Call a tool only when the user asked for something that tool serves and every required
   argument is known from the user's words or from a succeeded result. Use values the way
   the user stated them; do not invent values, years, or defaults the user did not give.
   Use identifiers exactly as they appear in tool results.
4. Multi-step requests: if a call needs a value from a result that does not exist yet,
   do not include it now. You will be asked again when results arrive; add it then.
5. Do not repeat a call that already succeeded unless the user explicitly asks to do it
   again. Do not call tools for small talk. When a request is clear, act on it without
   asking for confirmation.
6. Arguments must match the tool schema types. Omit optional arguments the user did not
   mention.
7. "reply" is spoken aloud: short, natural, 1-2 sentences.
   - Calls still pending: reply null, or a brief "progress" line that does not claim
     anything is done.
   - Every needed call succeeded: a "final" answer based only on the results.
   - A required detail is missing: a "clarify" question.
   - Small talk with no tool needed: a "final" reply.
   Never say an action is done unless its call has status "succeeded". If a call failed,
   say so honestly.
"""


class PlanParseError(ValueError):
    pass


# ── coercion ────────────────────────────────────────────────────────────────
_NUM_CLEAN = re.compile(r"[,\s$€£¥]")


def _to_number(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = _NUM_CLEAN.sub("", v.strip())
        try:
            return float(s)
        except ValueError:
            return v
    return v


def coerce_args(spec: ToolSpec, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce model-produced args to ``spec`` types. Returns (args, warnings)."""
    params = {p.name: p for p in spec.params}
    out: dict[str, Any] = {}
    warnings: list[str] = []
    for k, v in args.items():
        p = params.get(k)
        if p is None:
            warnings.append(f"{spec.name}: dropped unknown argument '{k}'")
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            if p.required:
                warnings.append(f"{spec.name}: empty required argument '{k}'")
            continue
        if p.type == "string":
            if isinstance(v, bool):
                v = "true" if v else "false"
            elif isinstance(v, float) and v.is_integer():
                v = str(int(v))
            elif isinstance(v, (int, float)):
                v = str(v)
            elif isinstance(v, str):
                v = v.strip()
            else:
                v = json.dumps(v)
        elif p.type in ("number", "integer"):
            n = _to_number(v)
            if isinstance(n, str) or isinstance(n, bool) or not isinstance(n, (int, float)):
                warnings.append(f"{spec.name}: could not read '{k}'={v!r} as a number")
            elif p.type == "integer" and float(n).is_integer():
                n = int(n)
            v = n
        elif p.type == "boolean" and isinstance(v, str):
            low = v.strip().lower()
            if low in ("true", "yes", "on", "1"):
                v = True
            elif low in ("false", "no", "off", "0"):
                v = False
            else:
                warnings.append(f"{spec.name}: could not read '{k}'={v!r} as a boolean")
        out[k] = v
    return out, warnings


# ── prompt / parsing ────────────────────────────────────────────────────────
def build_messages(snapshot: dict[str, Any], tools: Iterable[ToolSpec]) -> list[dict[str, str]]:
    schemas = [to_openai_tool(t)["function"] for t in tools]
    system = SYSTEM_PROMPT + "\nTools:\n" + json.dumps(schemas, ensure_ascii=False)
    payload = {
        "conversation": [
            {"role": "user" if c["role"] == "user" else "assistant", "text": c["text"]}
            for c in snapshot.get("conversation", [])
        ],
        "calls": [
            {"index": i, "tool": c["tool"], "args": c["args"], "status": c["status"],
             "result": c.get("result"), "error": c.get("error")}
            for i, c in enumerate(snapshot.get("calls", []))
        ],
    }
    if snapshot.get("orphan_effects"):
        payload["note"] = "These actions from a cancelled request completed anyway: " + json.dumps(
            snapshot["orphan_effects"], default=str)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


def parse_plan(raw: str | None, snapshot: dict[str, Any], specs: dict[str, ToolSpec]) -> tuple[Draft, list[str]]:
    try:
        obj = json.loads(_strip_fences(raw or ""))
    except json.JSONDecodeError as e:
        raise PlanParseError(f"model output is not JSON: {e}") from None
    if not isinstance(obj, dict):
        raise PlanParseError("model output is not a JSON object")

    warnings: list[str] = []
    existing = snapshot.get("calls", [])
    calls: list[ProposedCall] = []

    for i in obj.get("keep") or []:
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(existing):
            warnings.append(f"ignored invalid keep index {i!r}")
            continue
        c = existing[i]
        calls.append(ProposedCall(c["tool"], dict(c["args"]), occurrence=c.get("occurrence", 0)))

    for nc in obj.get("new_calls") or []:
        if not isinstance(nc, dict) or not isinstance(nc.get("tool"), str):
            warnings.append(f"ignored malformed call {nc!r}")
            continue
        spec = specs.get(nc["tool"])
        if spec is None:
            warnings.append(f"ignored call to unknown tool '{nc['tool']}'")
            continue
        raw_args = nc.get("args") if isinstance(nc.get("args"), dict) else {}
        args, w = coerce_args(spec, raw_args)
        warnings += w
        calls.append(ProposedCall(spec.name, args))

    reply = obj.get("reply")
    reply = reply.strip() if isinstance(reply, str) and reply.strip() else None
    kind = obj.get("reply_kind", "final")
    if kind not in ("final", "progress", "clarify"):
        warnings.append(f"unknown reply_kind {kind!r}; using 'final'")
        kind = "final"
    return Draft(tuple(calls), reply, kind), warnings


# ── worker ──────────────────────────────────────────────────────────────────
class OpenAIReasoner:
    def __init__(
        self,
        tools: Iterable[ToolSpec],
        client: Any = None,
        model: str | None = None,
        seed: int | None = 7,
        timeout_s: float = 20.0,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.tools = tuple(tools)
        self.specs = {t.name: t for t in self.tools}
        self.model = model or os.getenv("FDAGENT_REASONER_MODEL", DEFAULT_MODEL)
        self.seed = seed
        self.timeout_s = timeout_s
        self._client = client
        self._on_warning = on_warning or (lambda w: log.warning("reasoner: %s", w))

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI  # provider import stays out of core/runtime

            self._client = AsyncOpenAI()
        return self._client

    async def propose(self, request: RequestReasoning) -> Draft:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=build_messages(request.snapshot, self.tools),
            response_format={"type": "json_object"},
            temperature=0,
            timeout=self.timeout_s,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        resp = await self._get_client().chat.completions.create(**kwargs)
        draft, warnings = parse_plan(resp.choices[0].message.content, request.snapshot, self.specs)
        for w in warnings:
            self._on_warning(w)
        return draft
