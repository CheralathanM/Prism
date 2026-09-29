"""Tool backend protocol and schema export.

A backend executes one attempt of one operation. It may be synchronous (run in a worker
thread by the executor, so it can never block the event loop) or asynchronous (``acall``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fdagent.core.model import ToolSpec


class ToolTransientError(Exception):
    """A failure worth retrying. ``effect_applied`` says whether the state change happened
    (False = definitely not, None = unknown)."""

    def __init__(self, message: str, effect_applied: bool | None = None) -> None:
        super().__init__(message)
        self.effect_applied = effect_applied


@runtime_checkable
class SyncToolBackend(Protocol):
    def call(self, tool: str, args: dict[str, Any], idempotency_key: str) -> Any: ...


@runtime_checkable
class AsyncToolBackend(Protocol):
    async def acall(self, tool: str, args: dict[str, Any], idempotency_key: str) -> Any: ...


_JSON_TYPES = {"string": "string", "number": "number", "integer": "integer", "boolean": "boolean"}


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    """Chat Completions function-tool schema for a ToolSpec."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: {"type": _JSON_TYPES[p.type], "description": p.description} for p in spec.params
                },
                "required": [p.name for p in spec.params if p.required],
                "additionalProperties": False,
            },
        },
    }
