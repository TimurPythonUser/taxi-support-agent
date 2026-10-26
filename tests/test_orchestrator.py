"""The agent loop with a scripted model: control flow and deterministic guardrails."""

from taxi_agent.agent.state import SessionStatus
from taxi_agent.integrations.pricing import UnavailablePriceProvider
from taxi_agent.llm import LLMError

from .helpers import NOW, FakeLLM, call, say


async def start(agent):
    return (await agent.start_session()).session_id


async def test_lost_item_happy_path(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(
        call("update_case", scenario="lost_item", order_id="731842", item_description="чёрный зонт"),
        call("find_trip"),
        call("create_operator_ticket", summary="Клиент забыл чёрный зонт, заказ 731842"),
        say("Создал обращение OP-00001, оператор свяжется с вами."),
    )
    turn = await agent.handle_message(sid, "Забыл чёрный зонт, заказ 731842")

    assert [e.name for e in turn.events] == ["update_case", "find_trip", "create_operator_ticket"]
    assert turn.state.ticket_ids == ["OP-00001"]
    assert "OP-00001" in turn.reply
    _, history = await agent.store.load(sid)
    assert [m["role"] for m in history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_state_block_is_rerendered_after_each_tool(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(call("update_case", scenario="trip_price", origin="Крещатик 22"), say("Куда едем?"))
    await agent.handle_message(sid, "Сколько стоит доехать от Крещатика 22?")
    first_system, second_system = (req[0]["content"] for req in fake_llm.requests)
    assert "Active case: none yet" in first_system
    assert "origin: Крещатик 22" in second_system and "missing: destination" in second_system
    assert "Now: 2026-10-26 14:00 (Monday)" in second_system


async def test_human_request_is_handled_without_the_model(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    turn = await agent.handle_message(sid, "Соедините с оператором")
    assert fake_llm.requests == []
    assert turn.state.status == SessionStatus.HANDED_OFF
    assert [e.name for e in turn.events] == ["human_requested"]
    again = await agent.handle_message(sid, "алло?")
    assert "уже передано оператору" in again.reply and fake_llm.requests == []


async def test_invented_price_is_rewritten(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(say("Поездка будет стоить около 250 грн."), say("Подскажите адреса, и я посчитаю стоимость."))
    turn = await agent.handle_message(sid, "Сколько примерно до аэропорта?")
    assert turn.reply == "Подскажите адреса, и я посчитаю стоимость."
    assert any(e.name == "unverified_price" for e in turn.events)
    assert fake_llm.tool_flags == [True, False]


async def test_invented_price_twice_falls_back_to_safe_text(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(say("Около 250 грн."), say("Ну, примерно 260 грн."))
    turn = await agent.handle_message(sid, "Сколько примерно до аэропорта?")
    assert "250" not in turn.reply and "260" not in turn.reply
    assert "адрес" in turn.reply


async def test_quoted_price_passes_the_guard(agent, fake_llm: FakeLLM):
    sid = await start(agent)

    def answer_with_tool_price(messages):
        quote = next(m for m in reversed(messages) if m["role"] == "tool")
        import json

        price = json.loads(quote["content"])["quote"]["price"]
        return say(f"Предварительно {price} грн по тарифу «Эконом».")

    fake_llm.extend(
        call("update_case", scenario="trip_price", origin="Крещатик 22", destination="Аэропорт Борисполь"),
        call("get_trip_price"),
        answer_with_tool_price,
    )
    turn = await agent.handle_message(sid, "Сколько от Крещатика 22 до аэропорта Борисполь?")
    assert not any(e.kind == "guardrail" for e in turn.events)
    assert turn.state.active_case.quote.price == int(turn.reply.split()[1])


async def test_safety_incident_forces_urgent_handoff(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(say("Мне очень жаль. Подскажите номер заказа?"))
    turn = await agent.handle_message(sid, "Водитель был пьяный и чуть не попал в аварию")
    assert turn.state.status == SessionStatus.HANDED_OFF
    assert turn.state.handoff.priority == "urgent"
    assert "112" in turn.reply
    assert "ALERT" in fake_llm.requests[0][0]["content"]


async def test_model_handoff_gets_a_final_phrasing_round(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(
        call("handoff_to_operator", reason="вопрос о компенсации вне компетенции", priority="normal"),
        say("Передал ваш вопрос оператору, он ответит в этом чате."),
    )
    turn = await agent.handle_message(sid, "Хочу компенсацию за моральный ущерб")
    assert turn.state.status == SessionStatus.HANDED_OFF
    assert fake_llm.tool_flags == [True, False]
    assert turn.reply.startswith("Передал")


async def test_llm_errors_twice_lead_to_handoff(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(LLMError("timeout"), LLMError("timeout"))
    first = await agent.handle_message(sid, "Сколько стоит поездка?")
    assert "технический сбой" in first.reply and first.state.status == SessionStatus.ACTIVE
    second = await agent.handle_message(sid, "Сколько стоит поездка?")
    assert second.state.status == SessionStatus.HANDED_OFF


async def test_pricing_outage_twice_leads_to_handoff(settings):
    from taxi_agent.factory import build_agent

    fake = FakeLLM()
    agent = build_agent(settings, llm=fake, pricing=UnavailablePriceProvider(), clock=lambda: NOW)
    sid = await start(agent)
    fake.extend(
        call("update_case", scenario="trip_price", origin="Крещатик 22", destination="Вокзал"),
        call("get_trip_price"),
        say("Не получается рассчитать стоимость прямо сейчас, попробуйте через минуту."),
    )
    first = await agent.handle_message(sid, "Сколько от Крещатика 22 до вокзала?")
    assert first.state.status == SessionStatus.ACTIVE
    fake.extend(call("get_trip_price"), say("Сервис всё ещё недоступен."))
    second = await agent.handle_message(sid, "Попробуйте ещё раз")
    assert second.state.status == SessionStatus.HANDED_OFF
    assert any(e.name == "technical_handoff" for e in second.events)


async def test_no_progress_for_three_turns_hands_off(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(call("update_case", scenario="lost_item", item_description="сумка"), say("Когда была поездка?"))
    await agent.handle_message(sid, "Забыл сумку")
    for _ in range(3):
        fake_llm.extend(say("Подскажите, пожалуйста, дату поездки?"))
        turn = await agent.handle_message(sid, "не помню")
    assert turn.state.status == SessionStatus.HANDED_OFF
    assert any(e.name == "stalled_handoff" for e in turn.events)


async def test_session_message_limit(settings, fake_llm: FakeLLM):
    from taxi_agent.factory import build_agent

    settings.max_messages_per_session = 1
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    sid = await start(agent)
    fake_llm.extend(say("Здравствуйте!"))
    await agent.handle_message(sid, "Привет")
    turn = await agent.handle_message(sid, "Ещё вопрос")
    assert "лимит" in turn.reply and len(fake_llm.requests) == 1


async def test_history_window_cuts_at_user_turns(settings, fake_llm: FakeLLM):
    from taxi_agent.factory import build_agent

    settings.history_window_user_turns = 2
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    sid = await start(agent)
    for i in range(4):
        fake_llm.extend(call("search_knowledge_base", query="ожидание"), say(f"ответ {i}"))
        await agent.handle_message(sid, f"вопрос {i}")
    sent = fake_llm.requests[-1][1:]
    assert sent[0] == {"role": "user", "content": "вопрос 2"}
    tool_ids = {m["tool_call_id"] for m in sent if m["role"] == "tool"}
    call_ids = {tc["id"] for m in sent if m.get("tool_calls") for tc in m["tool_calls"]}
    assert tool_ids == call_ids  # no orphaned tool results


async def test_language_is_tracked(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(say("Hello! How can I help?"))
    turn = await agent.handle_message(sid, "Hi, I left my backpack in a taxi")
    assert turn.state.language == "en"
    assert "Customer language: en" in fake_llm.requests[0][0]["content"]
