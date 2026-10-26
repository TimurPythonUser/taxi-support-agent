"""One customer message in → one agent reply out.

deterministic pre-checks (handed off? limits? asks for a human? safety words?)
  → LLM with tools, up to N rounds (state block re-rendered every round)
  → deterministic post-checks (invented prices, forced safety handoff, no-progress handoff)
  → persist state + history
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from taxi_agent.agent import guardrails
from taxi_agent.agent.prompts import build_system_message
from taxi_agent.agent.state import Priority, SessionState, SessionStatus
from taxi_agent.agent.tools import TOOL_SCHEMAS, ToolContext, ToolExecutor, handoff
from taxi_agent.config import Settings
from taxi_agent.integrations.orders import OrderRepository
from taxi_agent.integrations.pricing import PriceProvider
from taxi_agent.integrations.tickets import TicketSink
from taxi_agent.kb import KnowledgeBase
from taxi_agent.llm import LLMClient, LLMError, Usage
from taxi_agent.storage import SQLiteStore

_PROGRESS_TOOLS = {"find_trip", "get_trip_price", "search_knowledge_base", "create_operator_ticket", "add_ticket_note"}


class SessionNotFound(LookupError):
    pass


@dataclass
class TurnEvent:
    kind: str  # tool | guardrail | error
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    ms: int = 0


@dataclass
class TurnResult:
    reply: str
    state: SessionState
    events: list[TurnEvent]


class Agent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        store: SQLiteStore,
        tickets: TicketSink,
        orders: OrderRepository,
        pricing: PriceProvider,
        kb: KnowledgeBase,
        clock=None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.tickets = tickets
        self.orders = orders
        self.pricing = pricing
        self.kb = kb
        self._clock = clock or (lambda: datetime.now(settings.tz))
        self._locks: dict[str, asyncio.Lock] = {}

    def now(self) -> datetime:
        return self._clock()

    async def start_session(self) -> SessionState:
        state = SessionState(session_id=uuid.uuid4().hex, created_at=self.now())
        await self.store.create(state)
        return state

    async def handle_message(self, session_id: str, text: str) -> TurnResult:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            loaded = await self.store.load(session_id)
            if loaded is None:
                raise SessionNotFound(session_id)
            state, messages = loaded
            result = await self._turn(state, messages, text.strip())
            await self.store.save(state, messages)
            return result

    # ------------------------------------------------------------------------------------------
    async def _turn(self, state: SessionState, messages: list[dict[str, Any]], text: str) -> TurnResult:
        events: list[TurnEvent] = []
        now = self.now()
        state.language = guardrails.detect_language(text, state.language)
        messages.append({"role": "user", "content": text})
        ctx = ToolContext(
            state=state,
            messages=messages,
            now=now,
            orders=self.orders,
            pricing=self.pricing,
            tickets=self.tickets,
            kb=self.kb,
        )

        def finish(reply: str) -> TurnResult:
            messages.append({"role": "assistant", "content": reply})
            return TurnResult(reply=reply, state=state, events=events)

        if state.status == SessionStatus.HANDED_OFF:
            events.append(TurnEvent("guardrail", "already_handed_off"))
            return finish(guardrails.fixed("handed_off", state.language))
        if state.user_turns >= self.settings.max_messages_per_session:
            events.append(TurnEvent("guardrail", "session_limit"))
            return finish(guardrails.fixed("session_limit", state.language))
        state.user_turns += 1

        if guardrails.wants_human(text):
            ticket_id = await handoff(ctx, "Клиент попросил соединить с оператором", Priority.NORMAL)
            events.append(TurnEvent("guardrail", "human_requested", result={"ticket_id": ticket_id}))
            return finish(guardrails.fixed("human_requested", state.language))

        if guardrails.is_safety_incident(text):
            state.safety_alert = True
            events.append(TurnEvent("guardrail", "safety_keywords"))

        if await self.store.count_request(now.date()) > self.settings.daily_request_limit:
            events.append(TurnEvent("guardrail", "daily_limit"))
            return finish(guardrails.fixed("daily_limit", state.language))

        before = state.missing_snapshot()
        failures_before = state.consecutive_failures
        executor = ToolExecutor(ctx)
        try:
            reply = await self._llm_loop(state, messages, executor, events)
        except LLMError as exc:
            events.append(TurnEvent("error", "llm", result={"error": str(exc)}))
            reply = None
        if reply is None:
            state.consecutive_failures += 1
        elif state.consecutive_failures == failures_before:
            state.consecutive_failures = 0  # a clean turn resets the streak

        if state.consecutive_failures >= 2 and state.status != SessionStatus.HANDED_OFF:
            ticket_id = await handoff(ctx, "Повторяющийся технический сбой (модель или внешний API)", Priority.NORMAL)
            events.append(TurnEvent("guardrail", "technical_handoff", result={"ticket_id": ticket_id}))
            return finish(guardrails.fixed("technical_handoff", state.language))
        if reply is None:
            return finish(guardrails.fixed("technical_error", state.language))

        reply = await self._check_prices(reply, state, messages, events)

        if state.safety_alert and state.status != SessionStatus.HANDED_OFF:
            ticket_id = await handoff(ctx, "Сообщение о возможной угрозе безопасности", Priority.URGENT)
            events.append(TurnEvent("guardrail", "safety_forced_handoff", result={"ticket_id": ticket_id}))
            reply = f"{reply}\n\n{guardrails.fixed('safety_forced', state.language)}"

        if state.status != SessionStatus.HANDED_OFF:
            after = state.missing_snapshot()
            stalled = before == after and bool(after[1]) and not (_PROGRESS_TOOLS & set(executor.ctx.calls))
            state.stalled_turns = state.stalled_turns + 1 if stalled else 0
            if state.stalled_turns >= self.settings.stall_turns_before_handoff:
                ticket_id = await handoff(ctx, "Агент не смог собрать данные за несколько попыток", Priority.NORMAL)
                events.append(TurnEvent("guardrail", "stalled_handoff", result={"ticket_id": ticket_id}))
                reply = guardrails.fixed("stalled", state.language)

        return finish(reply)

    async def _llm_loop(
        self,
        state: SessionState,
        messages: list[dict[str, Any]],
        executor: ToolExecutor,
        events: list[TurnEvent],
    ) -> str | None:
        """Run model ↔ tools rounds. Tool-call messages are appended to the persistent history."""
        for _ in range(self.settings.max_tool_iterations):
            response = await self._complete(state, messages)
            if not response.tool_calls:
                return (response.content or "").strip() or None
            messages.append(response.message)
            for call in response.tool_calls:
                started = time.perf_counter()
                result = await executor.run(call.name, call.arguments)
                try:
                    arguments = json.loads(call.arguments or "{}")
                except ValueError:
                    arguments = {"_raw": call.arguments}
                events.append(
                    TurnEvent(
                        "tool",
                        call.name,
                        arguments,
                        result,
                        round((time.perf_counter() - started) * 1000),
                    )
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": guardrails.tool_payload(result)})
            if state.status == SessionStatus.HANDED_OFF:
                # one more round so the model phrases the handoff message itself
                final = await self._complete(state, messages, tools=False)
                return (final.content or "").strip() or guardrails.fixed("handed_off", state.language)
        # Too many rounds: stop calling tools and let the model answer from what it already has.
        events.append(TurnEvent("guardrail", "tool_loop_limit"))
        final = await self._complete(state, messages, tools=False)
        return (final.content or "").strip() or None

    async def _complete(self, state: SessionState, messages: list[dict[str, Any]], tools: bool = True):
        system = build_system_message(state, self.now(), self.settings.company_name)
        response = await self.llm.complete([system, *self._window(messages)], TOOL_SCHEMAS if tools else [])
        self._account(state, response.usage)
        return response

    def _window(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Last N user turns. Cutting at a user message never splits a tool call from its result;
        older facts are not lost because they live in the case state block."""
        user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
        keep = self.settings.history_window_user_turns
        if len(user_positions) <= keep:
            return messages
        return messages[user_positions[-keep] :]

    @staticmethod
    def _account(state: SessionState, usage: Usage) -> None:
        state.prompt_tokens += usage.prompt_tokens
        state.completion_tokens += usage.completion_tokens
        state.cached_prompt_tokens += usage.cached_prompt_tokens

    async def _check_prices(
        self,
        reply: str,
        state: SessionState,
        messages: list[dict[str, Any]],
        events: list[TurnEvent],
    ) -> str:
        """Output guard: every money amount in the reply must come from a tool result."""
        invented = guardrails.unverified_amounts(reply, guardrails.grounded_numbers(messages))
        if not invented:
            return reply
        events.append(TurnEvent("guardrail", "unverified_price", result={"amounts": sorted(invented)}))
        correction = {
            "role": "system",
            "content": f"Your draft reply mentioned amounts {sorted(invented)} that no tool returned. "
            "Rewrite the reply without inventing any prices or amounts. Draft:\n" + reply,
        }
        try:
            response = await self._complete(state, [*messages, correction], tools=False)
        except LLMError:
            return guardrails.fixed("price_fallback", state.language)
        retry = (response.content or "").strip()
        grounded = guardrails.grounded_numbers(messages)
        if retry and not guardrails.unverified_amounts(retry, grounded):
            return retry
        return guardrails.fixed("price_fallback", state.language)
