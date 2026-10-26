"""Regressions found by code review of the QA fixes (see qa/PLAN.md)."""

import asyncio
import json

import pytest

from taxi_agent.agent import guardrails
from taxi_agent.agent.state import SessionStatus
from taxi_agent.factory import build_agent

from .helpers import NOW, FakeLLM, call, say


@pytest.mark.parametrize(
    "text",
    [
        "The driver charged me a very high price",
        "the driver's price was too high",
        "Мы сбились с дороги, водитель плохо знал город",
        "Водій збився з маршруту",
        "кровь из носа надо успеть к 9",
        "I left my bag in the car by accident",
    ],
)
def test_ordinary_complaints_are_not_safety_alarms(text):
    assert not guardrails.is_safety_incident(text)


@pytest.mark.parametrize(
    "text",
    [
        "Водитель наставил на меня пистолет",
        "Водій навів на мене пістолет",
        "Он достал из бардачка нож",
        "Водитель ко мне пристал",
        "Ми врізалися в стовп",
        "The driver was clearly high",
    ],
)
def test_threats_and_incidents_are_detected(text):
    assert guardrails.is_safety_incident(text)


@pytest.mark.parametrize(
    "text",
    ["Можно перевезти 5 человек?", "Скільки коштує перевезти 4 людини?", "не хочу ждать соедините с оператором"],
)
def test_human_request_edge_cases(text):
    expected = text.startswith("не хочу ждать")
    assert guardrails.wants_human(text) is expected


def test_why_not_connect_is_a_request():
    assert guardrails.wants_human("Почему не соединяете с оператором?")


@pytest.mark.parametrize(
    ("text", "previous", "expected"),
    [
        ("Сколько стоит до Васильківської?", "ru", "ru"),
        ("забыла iphone pro max", "ru", "ru"),
    ],
)
def test_language_markers_are_counted(text, previous, expected):
    assert guardrails.detect_language(text, previous) == expected


def test_price_phrases_with_colon_and_extra_words():
    assert guardrails.mentioned_amounts("Цена: 300") == {300}
    assert guardrails.mentioned_amounts("Стоимость поездки — 300") == {300}
    assert guardrails.mentioned_amounts("Цена от Крещатика 22 будет рассчитана") == set()


def test_amounts_in_words():
    assert guardrails.mentioned_amounts("дев'ятсот гривень") == {900}
    assert guardrails.mentioned_amounts("about seventy hryvnias") == {70}


def test_compensation_demands_are_not_customer_facts():
    assert guardrails.customer_amounts("Списали 480 грн, а показывало 245 грн") == {480, 245}
    assert guardrails.customer_amounts("Требую компенсацию 5000 грн") == set()


async def start(agent):
    return (await agent.start_session()).session_id


async def test_customer_price_guess_does_not_ground_the_reply(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(say("Да, 350 грн хватит."), say("Подскажите адрес отправления, и я посчитаю."))
    turn = await agent.handle_message(sid, "До Борисполя за 350 грн довезёте?")
    assert any(e.name == "unverified_price" for e in turn.events)
    assert "350" not in turn.reply


class BrokenTickets:
    async def create(self, draft):
        raise ConnectionError("helpdesk down")

    async def add_note(self, ticket_id, note):
        raise ConnectionError("helpdesk down")

    async def list(self, session_id=None):
        return []


async def test_ticket_system_outage_still_shows_112(settings, fake_llm: FakeLLM):
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    agent.tickets = BrokenTickets()
    sid = await start(agent)
    turn = await agent.handle_message(sid, "Водитель мне угрожал, позовите оператора")
    assert "112" in turn.reply
    assert any(e.kind == "error" and e.name == "safety_forced_handoff" for e in turn.events)


async def test_daily_limit_in_the_handoff_round_keeps_the_handoff_reply(settings, fake_llm: FakeLLM):
    settings.daily_request_limit = 1
    agent = build_agent(settings, llm=fake_llm, clock=lambda: NOW)
    sid = await start(agent)
    fake_llm.extend(call("handoff_to_operator", reason="вопрос вне базы", priority="normal"))
    turn = await agent.handle_message(sid, "Хочу компенсацию за моральный ущерб")
    assert turn.state.status == SessionStatus.HANDED_OFF
    assert "оператору" in turn.reply and "завтра" not in turn.reply


async def test_rejected_future_time_restores_previous_value(make_executor):
    ex = make_executor("вчера около 19:40")
    await ex.run(
        "update_case", json.dumps({"scenario": "lost_item", "trip_date": "2026-10-25", "approx_time": "19:40"})
    )
    result = await ex.run(
        "update_case", json.dumps({"scenario": "lost_item", "trip_date": "2026-10-26", "approx_time": "23:00"})
    )
    case = ex.ctx.state.active_case
    assert case.approx_time == "19:40" and str(case.trip_date) == "2026-10-25"
    assert {"trip_date", "approx_time"} <= set(result["rejected"])


async def test_note_can_target_an_earlier_ticket_explicitly(make_executor):
    ex = make_executor("заказ 731842")
    await ex.run(
        "update_case", json.dumps({"scenario": "lost_item", "order_id": "731842", "item_description": "сумка"})
    )
    await ex.run("find_trip", "{}")
    ticket = await ex.run("create_operator_ticket", json.dumps({"summary": "сумка"}))
    await ex.run("update_case", json.dumps({"scenario": "trip_price", "origin": "Крещатик 22"}))
    assert (await ex.run("add_ticket_note", json.dumps({"note": "в сумке паспорт"})))["ok"] is False
    explicit = await ex.run(
        "add_ticket_note", json.dumps({"note": "в сумке паспорт", "ticket_id": ticket["ticket_id"]})
    )
    assert explicit["ok"]
    assert (await ex.run("add_ticket_note", json.dumps({"note": "x", "ticket_id": "OP-99999"})))["ok"] is False


async def test_session_locks_are_released_under_concurrency(agent, fake_llm: FakeLLM):
    sid = await start(agent)
    fake_llm.extend(say("Ответ 1"), say("Ответ 2"), say("Ответ 3"))
    await asyncio.gather(*(agent.handle_message(sid, f"вопрос {i}") for i in range(3)))
    assert agent._locks == {}
