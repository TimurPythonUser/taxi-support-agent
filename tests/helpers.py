"""Test helpers: a scripted fake LLM and a fixed clock."""

import json
from collections.abc import Callable
from datetime import datetime
from itertools import count
from typing import Any
from zoneinfo import ZoneInfo

from taxi_agent.llm import LLMResponse, ToolCall

KYIV = ZoneInfo("Europe/Kyiv")
NOW = datetime(2026, 10, 26, 14, 0, tzinfo=KYIV)  # Monday; fixtures: "yesterday" = 2026-10-25

Step = LLMResponse | Exception | Callable[[list[dict[str, Any]]], LLMResponse]
_ids = count(1)


def call(name: str, **arguments: Any) -> LLMResponse:
    """Scripted model turn that requests one tool call."""
    tc = ToolCall(id=f"call_{next(_ids)}", name=name, arguments=json.dumps(arguments, ensure_ascii=False))
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}],
    }
    return LLMResponse(content=None, tool_calls=[tc], message=message)


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text, message={"role": "assistant", "content": text})


class FakeLLM:
    """Replays scripted responses and records what the agent sent."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_flags: list[bool] = []

    def extend(self, *steps: Step) -> None:
        self.steps.extend(steps)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.requests.append(messages)
        self.tool_flags.append(bool(tools))
        if not self.steps:
            raise AssertionError("FakeLLM: unexpected extra LLM call")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, LLMResponse):
            return step(messages)
        return step
