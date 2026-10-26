"""Agent loop, tools and HTTP API with a scripted model. IDs match qa/PLAN.md."""

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from taxi_agent.api.app import create_app
from taxi_agent.config import Settings
from taxi_agent.factory import build_agent
from taxi_agent.integrations.pricing import PriceQuote, UnavailablePriceProvider
from taxi_agent.kb import KnowledgeBase
from taxi_agent.llm import LLMResponse, ToolCall
from tests.helpers import NOW, call, say


def args(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


def raw_call(name: str, arguments: str) -> LLMResponse:
    """Model turn with tool arguments exactly as given (not necessarily a JSON object)."""
    tc = ToolCall(id="call_raw", name=name, arguments=arguments)
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    }
    return LLMResponse(content=None, tool_calls=[tc], message=message)


async def start(agent) -> str:
    return (await agent.start_session()).session_id


class FixedPrice:
    def __init__(self, price: int) -> None:
        self.price = price

    async def quote(self, origin: str, destination: str, tariff: str) -> PriceQuote:
        return PriceQuote(
            quote_id="Q-test",
            origin=origin,
            destination=destination,
            tariff=tariff,
            distance_km=24.0,
            duration_min=65,
            price=self.price,
            currency="UAH",
            valid_until=NOW + timedelta(minutes=10),
        )


class BrokenOrders:
    """A real order API that times out."""

    async def get(self, order_id):
        raise ConnectionError("orders API timeout")

    async def search(self, trip_date, approx_time, route):
        raise ConnectionError("orders API timeout")


# --- CRIT --------------------------------------------------------------------------------------


@pytest.mark.crit
async def test_c01_safety_report_after_a_normal_handoff_is_escalated(agent, fake_llm):
    """C-01: once handed off, the session answers with a fixed text BEFORE the safety check runs.
    A later "the driver is drunk" stays a NORMAL ticket; at night only urgent ones are handled."""
    sid = await start(agent)
    await agent.handle_message(sid, "Соедините с оператором")
    turn = await agent.handle_message(sid, "Водитель пьяный, гонит, мне страшно")
    tickets = await agent.tickets.list(sid)
    assert turn.state.safety_alert
    assert turn.state.handoff.priority == "urgent" or any(t.priority == "urgent" for t in tickets)


# --- MAJOR -------------------------------------------------------------------------------------


@pytest.mark.major
async def test_m01_wrong_order_number_does_not_block_search_by_date_and_time(make_executor):
    """M-01: find_trip says "ask for date and time", but while order_id is set the lookup is always by
    order number, the date/time never invalidate it and the model cannot clear order_id."""
    ex = make_executor(
        "Забыл зонт, заказ 123456",
        "Ой, номер не тот. Ехал вчера около 19:40 с Крещатика на вокзал",
    )
    await ex.run("update_case", args(scenario="lost_item", order_id="123456", item_description="зонт"))
    first = await ex.run("find_trip", "{}")
    assert first["matches"] == []
    await ex.run(
        "update_case",
        args(scenario="lost_item", trip_date="2026-10-25", approx_time="19:40", route="Крещатик → вокзал"),
    )
    second = await ex.run("find_trip", "{}")
    assert [m["order_id"] for m in second["matches"]] == ["731842"]


@pytest.mark.major
async def test_m02_english_price_over_1000_reaches_the_customer(settings, fake_llm):
    """M-02: P-01 end to end — a correct Business quote in English is replaced by "I can't name a price"."""
    agent = build_agent(settings, llm=fake_llm, pricing=FixedPrice(1150), clock=lambda: NOW)
    sid = await start(agent)
    answer = "A Business ride is 1,150 UAH (estimate, valid for 10 minutes)."
    fake_llm.extend(
        call(
            "update_case",
            scenario="trip_price",
            origin="Obolonskyi avenue 15",
            destination="Boryspil airport",
            tariff="business",
        ),
        call("get_trip_price"),
        say(answer),
        say(answer),  # rewrite round, used only if the guard misfires
    )
    turn = await agent.handle_message(sid, "How much is a Business ride from Obolonskyi avenue 15 to Boryspil airport?")
    assert "1,150" in turn.reply
    assert not any(e.name == "unverified_price" for e in turn.events)


@pytest.mark.major
async def test_m03_amount_typed_by_the_customer_is_not_treated_as_invented(agent, fake_llm):
    """M-03: only tool results ground amounts. Repeating the customer's own "480 грн" in a fare complaint
    ends with the price fallback "tell me the pickup and drop-off addresses"."""
    sid = await start(agent)
    reply = "Понимаю: показывало 245 грн, а списали 480 грн. Подскажите, что именно сделал водитель?"
    fake_llm.extend(
        call("update_case", scenario="driver_complaint", order_id="731842", reason_category="route_or_fare"),
        call("find_trip"),
        say(reply),
        say(reply),
    )
    text = "Заказ 731842: показывало 245 грн, а списали 480 грн, водитель катал кругами"
    turn = await agent.handle_message(sid, text)
    assert "480" in turn.reply
    assert "адрес" not in turn.reply


@pytest.mark.major
async def test_m04_note_for_a_new_complaint_does_not_land_in_the_lost_item_ticket(make_executor):
    """M-04: add_ticket_note falls back to the LAST ticket of the session, even when the active case is a
    different, still collecting case → complaint details end up in the lost-item ticket."""
    ex = make_executor("Забыл рюкзак, заказ 731842", "И ещё: водитель грубил всю дорогу")
    await ex.run("update_case", args(scenario="lost_item", order_id="731842", item_description="рюкзак"))
    await ex.run("find_trip", "{}")
    lost = await ex.run("create_operator_ticket", args(summary="рюкзак"))
    await ex.run("update_case", args(scenario="driver_complaint", order_id="731842", reason_category="rude_behavior"))
    note = await ex.run("add_ticket_note", args(note="Водитель грубил всю дорогу"))
    [ticket] = [t for t in await ex.ctx.tickets.list("s1") if t.id == lost["ticket_id"]]
    assert ticket.notes == []
    assert note["ok"] is False


@pytest.mark.major
@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("чёрный зонт, лежал справа на заднем сиденье", "normal"),  # "справа" contains "права"
        ("очки, оправа золотистая", "normal"),
        ("картонная коробка с книгами", "normal"),  # "карт"
        ("айфон в чёрном чехле", "high"),
        ("смартфон Samsung", "high"),
        ("наушники AirPods Pro", "high"),
        ("MacBook Air серый", "high"),
        ("кредитка Монобанка", "high"),
        ("бумажник с наличными", "high"),
    ],
)
async def test_m05_lost_item_priority_matches_the_policy(make_executor, item, expected):
    """M-05: the valuables regex has no word boundaries and misses common valuables.
    The tool schema itself asks for "where in the car", so "справа/слева" is everywhere."""
    ex = make_executor("заказ 731842")
    await ex.run("update_case", args(scenario="lost_item", order_id="731842", item_description=item))
    await ex.run("find_trip", "{}")
    result = await ex.run("create_operator_ticket", args(summary=item))
    assert result["priority"] == expected


@pytest.mark.major
async def test_m06_lost_knife_is_a_lost_item_not_an_emergency(agent, fake_llm):
    """M-06: S-01 end to end — a normal lost-item ticket plus an URGENT handoff and a frozen session."""
    sid = await start(agent)
    fake_llm.extend(
        call("update_case", scenario="lost_item", order_id="731842", item_description="складной нож в чехле"),
        call("find_trip"),
        call("create_operator_ticket", summary="Клиент забыл складной нож, заказ 731842"),
        say("Создал обращение OP-00001, оператор свяжется с водителем."),
    )
    turn = await agent.handle_message(sid, "Забыл в такси складной нож в чехле, заказ 731842")
    assert turn.state.status == "active"
    assert {t.priority for t in await agent.tickets.list(sid)} == {"normal"}


@pytest.mark.major
async def test_m07_order_service_outage_gives_a_reply_not_a_crash(settings, fake_llm):
    """M-07: only LLMError is caught. A real integration raising ConnectionError → HTTP 500,
    the customer's message is not saved."""
    agent = build_agent(settings, llm=fake_llm, orders=BrokenOrders(), clock=lambda: NOW)
    sid = await start(agent)
    fake_llm.extend(
        call("update_case", scenario="lost_item", order_id="731842", item_description="зонт"),
        call("find_trip"),
        say("Сервис поиска поездок сейчас недоступен, попробуйте чуть позже."),
    )
    try:
        turn = await agent.handle_message(sid, "Забыл зонт, заказ 731842")
    except Exception as exc:
        pytest.fail(f"integration error escaped the agent: {exc!r}")
    assert turn.reply
    _, history = await agent.store.load(sid)
    assert history and history[0]["content"] == "Забыл зонт, заказ 731842"


@pytest.mark.major
def test_m08_non_object_tool_arguments_do_not_break_the_http_turn(settings, fake_llm):
    """M-08: arguments "null" are kept in TurnEvent as None, EventOut(arguments: dict) fails → HTTP 500,
    although the turn was already saved. The UI says "server unreachable" while the agent answered."""
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    fake_llm.extend(raw_call("find_trip", "null"), say("Подскажите, пожалуйста, номер заказа?"))
    with TestClient(create_app(settings=settings, agent=agent), raise_server_exceptions=False) as client:
        sid = client.post("/api/sessions").json()["session_id"]
        resp = client.post(f"/api/sessions/{sid}/messages", json={"text": "Забыл зонт"})
    assert resp.status_code == 200


@pytest.mark.major
async def test_m09_model_cannot_ground_its_own_price_through_a_case_field(agent, fake_llm):
    """M-09: update_case echoes the case back as a tool result, so any amount written into a free-text
    field becomes "grounded" — the model can bypass the output guard by itself (found in the live run)."""
    sid = await start(agent)
    fake_llm.extend(
        call("update_case", scenario="trip_price", origin="Крещатик 22", destination="вокзал, обычно около 300 грн"),
        say("До вокзала поездка будет стоить около 300 грн."),
        say("Сейчас посчитаю стоимость."),
    )
    turn = await agent.handle_message(sid, "Сколько стоит от Крещатика 22 до вокзала?")
    assert any(e.name == "unverified_price" for e in turn.events)


# --- MINOR -------------------------------------------------------------------------------------


@pytest.mark.minor
def test_n01_whitespace_only_message_is_rejected(settings, fake_llm):
    """N-01: min_length=1 counts spaces; the message is stripped to "" and still goes to the model."""
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    fake_llm.extend(say("Здравствуйте! Чем помочь?"))
    with TestClient(create_app(settings=settings, agent=agent)) as client:
        sid = client.post("/api/sessions").json()["session_id"]
        resp = client.post(f"/api/sessions/{sid}/messages", json={"text": "   \n  "})
    assert resp.status_code == 422
    assert fake_llm.requests == []


@pytest.mark.minor
async def test_n02_ticket_waits_until_the_customer_picks_one_of_several_trips(make_executor):
    """N-02: next_step says "ask which trip", but create_operator_ticket does not enforce it."""
    ex = make_executor("вчера около 20:00, забыл зонт")
    await ex.run(
        "update_case",
        args(scenario="lost_item", trip_date="2026-10-25", approx_time="20:00", item_description="зонт"),
    )
    found = await ex.run("find_trip", "{}")
    assert len(found["matches"]) == 2
    result = await ex.run("create_operator_ticket", args(summary="зонт"))
    assert result["ok"] is False


@pytest.mark.minor
async def test_n03_one_safety_incident_gives_one_urgent_ticket(agent, fake_llm):
    """N-03: complaint ticket (urgent) + forced safety handoff ticket (urgent) for the same incident."""
    sid = await start(agent)
    fake_llm.extend(
        call(
            "update_case",
            scenario="driver_complaint",
            order_id="731655",
            reason_category="dangerous_driving",
            description="Водитель гнал на красный и превышал скорость",
            safety_incident=True,
        ),
        call("find_trip"),
        call("create_operator_ticket", summary="Опасное вождение, заказ 731655"),
        say("Создал срочное обращение OP-00001. Если есть угроза жизни, звоните 112."),
    )
    await agent.handle_message(sid, "Заказ 731655: водитель гнал на красный и превышал скорость, было страшно")
    urgent = [t for t in await agent.tickets.list(sid) if t.priority == "urgent"]
    assert len(urgent) == 1


@pytest.mark.minor
async def test_n04_one_retry_inside_a_turn_is_not_a_repeated_outage(settings, fake_llm):
    """N-04: two failed get_trip_price calls in ONE turn → "the problem keeps happening" handoff
    on the very first customer message."""
    agent = build_agent(settings, llm=fake_llm, pricing=UnavailablePriceProvider(), clock=lambda: NOW)
    sid = await start(agent)
    fake_llm.extend(
        call("update_case", scenario="trip_price", origin="Крещатик 22", destination="Вокзал"),
        call("get_trip_price"),
        call("get_trip_price"),
        say("Сервис расчёта сейчас недоступен, попробуйте через минуту."),
    )
    turn = await agent.handle_message(sid, "Сколько от Крещатика 22 до вокзала?")
    assert turn.state.status == "active"


@pytest.mark.minor
async def test_n05_daily_limit_bounds_model_calls_not_just_turns(settings, fake_llm):
    """N-05: the "daily model request limit" counts customer turns; one turn may make up to ~10 model calls."""
    settings.daily_request_limit = 2
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    sid = await start(agent)
    fake_llm.extend(
        call("search_knowledge_base", query="ожидание"),
        call("search_knowledge_base", query="платное ожидание"),
        say("Первые 3 минуты ожидания бесплатны."),
    )
    await agent.handle_message(sid, "Ожидание платное?")
    assert len(fake_llm.requests) <= settings.daily_request_limit


@pytest.mark.minor
async def test_n06_history_stays_bounded_after_the_session_limit(settings, fake_llm):
    """N-06: over the limit every message is still stored (user + fixed reply); only handed-off
    sessions are capped at 400 messages."""
    settings.max_messages_per_session = 1
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    sid = await start(agent)
    fake_llm.extend(say("Здравствуйте!"))
    await agent.handle_message(sid, "Привет")
    for i in range(250):
        await agent.handle_message(sid, f"спам {i}")
    _, history = await agent.store.load(sid)
    assert len(history) <= 400


@pytest.mark.minor
async def test_n07_trip_later_today_is_rejected_as_future(make_executor):
    """N-07: only the date is checked against "now"; today at 23:00 is accepted at 14:00."""
    ex = make_executor("сегодня в 23:00")
    result = await ex.run("update_case", args(scenario="lost_item", trip_date="2026-10-26", approx_time="23:00"))
    assert {"trip_date", "approx_time"} & set(result.get("rejected", {}))


@pytest.mark.minor
async def test_n08_contact_phone_must_come_from_the_customer(make_executor):
    """N-08: order_id is grounded in the customer's messages, the phone the operator will call is not."""
    ex = make_executor("Забыл зонт, заказ 731842")
    result = await ex.run("update_case", args(scenario="lost_item", contact_phone="+380 67 123 45 67"))
    assert "contact_phone" in result.get("rejected", {})


@pytest.mark.minor
async def test_n09_phone_number_is_not_accepted_as_an_order_number(make_executor):
    """N-09: a 10-digit phone typed in one run passes the 5–10 digit order-number check."""
    ex = make_executor("Мой номер 0671234567, перезвоните")
    result = await ex.run("update_case", args(scenario="lost_item", order_id="0671234567"))
    assert "order_id" in result.get("rejected", {})


@pytest.mark.minor
async def test_n10_old_trip_fare_cannot_be_passed_off_as_a_new_quote(agent, fake_llm):
    """N-10: grounding is global: the fare of a found trip (245) "confirms" any price in any context."""
    sid = await start(agent)
    fake_llm.extend(
        call("update_case", scenario="lost_item", order_id="731842", item_description="зонт"),
        call("find_trip"),
        say("Нашёл поездку 731842. До аэропорта Борисполь поездка будет стоить 245 грн."),
        say("Нашёл поездку 731842."),
    )
    turn = await agent.handle_message(sid, "Забыл зонт, заказ 731842. И сколько будет стоить до Борисполя?")
    assert any(e.name == "unverified_price" for e in turn.events)


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(Settings(_env_file=None).knowledge_base_dir)


@pytest.mark.minor
@pytest.mark.parametrize(
    ("query", "section"),
    [
        ("cancellation fee", "Отмена заказа"),
        ("скільки коштує скасування замовлення", "Отмена заказа"),
        ("можно ли перевезти кота", "Детское кресло и животные"),
        ("водитель взял больше, чем показывало приложение", "Неправильный маршрут или завышенная стоимость"),
    ],
)
def test_n11_kb_finds_the_section_among_the_three_results(kb, query, section):
    """N-11: BM25 + Russian stemmer: no hits for uk/en queries and for synonyms → "offer an operator"."""
    assert section in [chunk.section for chunk, _ in kb.search(query)]
