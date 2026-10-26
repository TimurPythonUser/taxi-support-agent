"""Thin LLM layer: one Protocol, one OpenAI-compatible implementation.

DeepSeek, OpenAI, Gemini (OpenAI endpoint), vLLM, Ollama etc. all speak Chat Completions,
so switching the provider is a `.env` change: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY / LLM_EXTRA_BODY.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import openai
from openai import AsyncOpenAI

from taxi_agent.config import Settings


class LLMError(RuntimeError):
    """The model could not be reached or returned an unusable response."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_prompt_tokens += other.cached_prompt_tokens


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # Assistant message exactly as it must be appended to the history
    # (keeps provider extras such as DeepSeek's reasoning_content when thinking is on).
    message: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class OpenAICompatClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY (or DEEPSEEK_API_KEY) is not set")
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        s = self._settings
        try:
            response = await self._client.chat.completions.create(
                model=s.llm_model,
                messages=messages,
                tools=tools or openai.omit,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
                extra_body=s.llm_extra_body or None,
            )
        except openai.APIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if not response.choices:
            raise LLMError("LLM returned no choices")
        msg = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        ]

        message: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            message["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in tool_calls
            ]
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            message["reasoning_content"] = reasoning

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                cached_prompt_tokens=getattr(response.usage, "prompt_cache_hit_tokens", 0) or 0,
            )
        return LLMResponse(content=msg.content, tool_calls=tool_calls, usage=usage, message=message)
