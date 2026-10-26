"""One customer message in → one agent reply out.

deterministic pre-checks (handed off? limits? asks for a human? safety words?)
  → LLM with tools, up to N rounds (state block re-rendered every round)
  → deterministic post-checks (invented prices, forced safety handoff, no-progress handoff)
  → persist state + history
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from taxi_agent.agent import guardrails
from taxi_agent.agent.prompts import build_system_message
from taxi_agent.agent.state import DriverComplaintCase, Priority, SessionState, SessionStatus, TripPriceCase
from taxi_agent.agent.tools import TOOL_SCHEMAS, ToolContext, ToolExecutor, escalate_to_urgent, handoff
from taxi_agent.config import Settings
from taxi_agent.integrations.orders import OrderRepository
from taxi_agent.integrations.pricing import PriceProvider
from taxi_agent.integrations.tickets import TicketSink
from taxi_agent.kb import KnowledgeBase
from taxi_agent.llm import LLMClient, LLMError, Usage
from taxi_agent.storage import SQLiteStore

logger = logging.getLogger(__name__)

_MAX_FROZEN_HISTORY = 400  # messages kept for a handed-off session; later pings are answered but not stored
_PROGRESS_TOOLS = {"find_trip", "get_trip_price", "search_knowledge_base", "create_operator_ticket", "add_ticket_note"}


class SessionNotFound(LookupError):
    pass


class DailyLimitReached(RuntimeError):
    """The demo's daily budget of model calls is used up."""


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
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}  # session_id -> (lock, requests holding it)

    def now(self) -> datetime:
        return self._clock()

    async def start_session(self) -> SessionState:
        state = SessionState(session_id=uuid.uuid4().hex, created_at=self.now())
        await self.store.create(state)
        return state

    async def handle_message(self, session_id: str, text: str) -> TurnResult:
        if await self.store.load(session_id) is None:  # no lock objects for unknown ids
            raise SessionNotFound(session_id)
        lock, users = self._locks.get(session_id, (asyncio.Lock(), 0))
        self._locks[session_id] = (lock, users + 1)
        try:
            async with lock:
                loaded = await self.store.load(session_id)
                if loaded is None:
                    raise SessionNotFound(session_id)
                state, messages = loaded
                result = await self._turn(state, messages, text.strip())
                await self.store.save(state, messages)
                return result
        finally:
            # drop the lock once nobody waits on it, so memory does not grow with every session
            lock, users = self._locks[session_id]
            if users <= 1:
                del self._locks[session_id]
            else:
                self._locks[session_id] = (lock, users - 1)

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

        def frozen(reply: str, event: str) -> TurnResult:
            """Fixed reply for a session that no longer talks to the model; its stored history stays bounded."""
            if len(messages) > _MAX_FROZEN_HISTORY:
                messages.pop()
                return TurnResult(reply=reply, state=state, events=events)
            events.append(TurnEvent("guardrail", event))
            return finish(reply)

        async def safe(action, event: str, *args) -> str | None:
            """Run handoff/escalation; if the ticket system fails, log it and still answer the customer."""
            try:
                ticket_id = await action(ctx, *args)
            except Exception as exc:
                logger.exception("%s failed", event)
                events.append(TurnEvent("error", event, result={"error": type(exc).__name__}))
                return None
            events.append(TurnEvent("guardrail", event, result={"ticket_id": ticket_id}))
            return ticket_id

        async def safety_handoff(reply: str | None = None) -> TurnResult:
            await safe(handoff, "safety_forced_handoff", "Сообщение о возможной угрозе безопасности", Priority.URGENT)
            notice = guardrails.fixed("safety_forced", state.language)
            return finish(f"{reply}\n\n{notice}" if reply else notice)

        # Safety first: it must win over every other early exit (handoff, limits, human request, model errors).
        safety = guardrails.is_safety_incident(text)
        if safety:
            state.safety_alert = True
            events.append(TurnEvent("guardrail", "safety_keywords"))

        if state.status == SessionStatus.HANDED_OFF:
            if safety:
                if not (state.handoff and state.handoff.priority == Priority.URGENT):
                    reason = "Сообщение об угрозе безопасности после передачи оператору"
                    await safe(escalate_to_urgent, "safety_escalated", reason)
                return frozen(guardrails.fixed("safety_forced", state.language), "safety_after_handoff")
            return frozen(guardrails.fixed("handed_off", state.language), "already_handed_off")

        if state.user_turns >= self.settings.max_messages_per_session:
            if state.safety_alert:
                return await safety_handoff()
            return frozen(guardrails.fixed("session_limit", state.language), "session_limit")
        state.user_turns += 1

        if guardrails.wants_human(text):
            if state.safety_alert:
                return await safety_handoff()
            await safe(handoff, "human_requested", "Клиент попросил соединить с оператором", Priority.NORMAL)
            return finish(guardrails.fixed("human_requested", state.language))

        before = state.missing_snapshot()
        executor = ToolExecutor(ctx)
        try:
            if await self.store.request_count(now.date()) >= self.settings.daily_request_limit:
                raise DailyLimitReached
            reply = await self._llm_loop(state, messages, executor, events)
        except DailyLimitReached:
            if state.safety_alert and state.status != SessionStatus.HANDED_OFF:
                return await safety_handoff()
            events.append(TurnEvent("guardrail", "daily_limit"))
            if state.status == SessionStatus.HANDED_OFF:  # handed off mid-turn: never answer "come back tomorrow"
                key = "safety_forced" if state.safety_alert else "handed_off"
                return finish(guardrails.fixed(key, state.language))
            return finish(guardrails.fixed("daily_limit", state.language))
        except LLMError as exc:
            events.append(TurnEvent("error", "llm", result={"error": str(exc)}))
            reply = None
        except Exception as exc:  # a bug or an integration failure must not lose the customer's message
            logger.exception("agent turn failed")
            events.append(TurnEvent("error", "internal", result={"error": type(exc).__name__}))
            reply = None
        # one failure per turn at most: a retry inside the turn is not "the problem keeps happening"
        if reply is None or executor.ctx.turn_failed:
            state.consecutive_failures += 1
        else:
            state.consecutive_failures = 0

        if reply is None and state.status == SessionStatus.HANDED_OFF:
            # the model handed off but failed to phrase the final message
            reply = guardrails.fixed("handed_off", state.language)
        if state.safety_alert and state.status != SessionStatus.HANDED_OFF and reply is None:
            return await safety_handoff()
        if state.consecutive_failures >= 2 and state.status != SessionStatus.HANDED_OFF:
            reason = "Повторяющийся технический сбой (модель или внешний API)"
            await safe(handoff, "technical_handoff", reason, Priority.NORMAL)
            return finish(guardrails.fixed("technical_handoff", state.language))
        if reply is None:
            return finish(guardrails.fixed("technical_error", state.language))

        reply = await self._check_prices(reply, state, messages, events)

        if state.safety_alert and state.status != SessionStatus.HANDED_OFF:
            return await safety_handoff(reply)

        if state.status != SessionStatus.HANDED_OFF:
            after = state.missing_snapshot()
            stalled = before == after and bool(after[1]) and not (_PROGRESS_TOOLS & set(executor.ctx.calls))
            state.stalled_turns = state.stalled_turns + 1 if stalled else 0
            if state.stalled_turns >= self.settings.stall_turns_before_handoff:
                reason = "Агент не смог собрать данные за несколько попыток"
                await safe(handoff, "stalled_handoff", reason, Priority.NORMAL)
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
                    arguments = None
                if not isinstance(arguments, dict):  # "null", "[]" or broken JSON from the model
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
                try:
                    final = await self._complete(state, messages, tools=False)
                except (LLMError, DailyLimitReached):
                    key = "safety_forced" if state.safety_alert else "handed_off"
                    return guardrails.fixed(key, state.language)
                return (final.content or "").strip() or guardrails.fixed("handed_off", state.language)
        # Too many rounds: stop calling tools and let the model answer from what it already has.
        events.append(TurnEvent("guardrail", "tool_loop_limit"))
        final = await self._complete(state, messages, tools=False)
        return (final.content or "").strip() or None

    async def _complete(self, state: SessionState, messages: list[dict[str, Any]], tools: bool = True):
        # the demo cost guard counts model calls, not customer turns (one turn may make several calls)
        if await self.store.count_request(self.now().date()) > self.settings.daily_request_limit:
            raise DailyLimitReached
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
        """Output guard: every money amount in the reply must come from a quote, the knowledge base or —
        outside price quotes — from the customer's own words."""
        include_customer = isinstance(state.active_case, DriverComplaintCase)  # "списали 480 грн"
        grounded = guardrails.grounded_numbers(messages, include_customer=include_customer)
        invented = guardrails.unverified_amounts(reply, grounded)
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
        except (LLMError, DailyLimitReached):
            retry = ""
        else:
            retry = (response.content or "").strip()
        if retry and not guardrails.unverified_amounts(retry, grounded):
            return retry
        price_talk = state.active_case is None or isinstance(state.active_case, TripPriceCase)
        return guardrails.fixed("price_fallback" if price_talk else "amount_fallback", state.language)
