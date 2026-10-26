"""Tool handlers: validation, grounding and the rules that keep the model honest."""

import pytest

from taxi_agent.agent.state import CaseStatus, Scenario, SessionState
from taxi_agent.agent.tools import ToolContext, ToolExecutor
from taxi_agent.config import Settings
from taxi_agent.integrations.orders import StubOrderRepository
from taxi_agent.integrations.pricing import StubPriceProvider, UnavailablePriceProvider
from taxi_agent.kb import KnowledgeBase
from taxi_agent.storage import SQLiteStore
from taxi_agent.storage.sqlite import SQLiteTicketSink

from .helpers import KYIV, NOW


@pytest.fixture
def make_executor(tmp_path):
    kb = KnowledgeBase(Settings(_env_file=None).knowledge_base_dir)
    tickets = SQLiteTicketSink(SQLiteStore(tmp_path / "t.db"))

    def make(*user_messages: str, pricing=None) -> ToolExecutor:
        ctx = ToolContext(
            state=SessionState(session_id="s1", created_at=NOW),
            messages=[{"role": "user", "content": m} for m in user_messages],
            now=NOW,
            orders=StubOrderRepository(KYIV, "UAH", now=NOW),
            pricing=pricing or StubPriceProvider("UAH"),
            tickets=tickets,
            kb=kb,
        )
        return ToolExecutor(ctx)

    return make


def args(**kwargs) -> str:
    import json

    return json.dumps(kwargs, ensure_ascii=False)


class TestUpdateCase:
    async def test_order_id_the_customer_never_typed_is_rejected(self, make_executor):
        ex = make_executor("Я забыл сумку")
        result = await ex.run("update_case", args(scenario="lost_item", order_id="731842"))
        assert "order_id" in result["rejected"]
        assert ex.ctx.state.active_case.order_id is None

    async def test_order_id_typed_with_separators_is_accepted(self, make_executor):
        ex = make_executor("заказ №731-842")
        result = await ex.run("update_case", args(scenario="lost_item", order_id="731842"))
        assert result["applied"] == {"order_id": "731842"}
        assert result["next_step"].startswith("call find_trip")

    async def test_order_id_from_find_trip_candidates_is_accepted(self, make_executor):
        ex = make_executor("вчера около 20:00, забыл зонт")
        await ex.run(
            "update_case",
            args(scenario="lost_item", trip_date="2026-10-25", approx_time="20:00", item_description="зонт"),
        )
        found = await ex.run("find_trip", "{}")
        assert len(found["matches"]) == 2
        assert "which of the matching trips" in found["next_step"]
        picked = await ex.run("update_case", args(scenario="lost_item", order_id="731870"))
        assert picked["applied"] == {"order_id": "731870"}

    @pytest.mark.parametrize(
        ("field", "value", "fragment"),
        [
            ("trip_date", "2026-10-27", "future"),
            ("trip_date", "вчера", "YYYY-MM-DD"),
            ("trip_date", "2026-01-01", "90 days"),
            ("approx_time", "вечером", "HH:MM"),
        ],
    )
    async def test_invalid_values_are_rejected_with_a_hint(self, make_executor, field, value, fragment):
        ex = make_executor("...")
        result = await ex.run("update_case", args(scenario="lost_item", **{field: value}))
        assert fragment in result["rejected"][field]

    async def test_time_is_normalized(self, make_executor):
        ex = make_executor("...")
        result = await ex.run("update_case", args(scenario="lost_item", approx_time="8.05"))
        assert result["applied"]["approx_time"] == "08:05"

    async def test_fields_of_other_scenarios_are_ignored(self, make_executor):
        ex = make_executor("...")
        result = await ex.run("update_case", args(scenario="trip_price", origin="Крещатик 22", item_description="x"))
        assert result["ignored_fields"] == ["item_description"]

    async def test_cannot_confirm_a_trip_that_was_not_found(self, make_executor):
        ex = make_executor("...")
        result = await ex.run("update_case", args(scenario="lost_item", trip_confirmed=True))
        assert "trip_confirmed" in result["rejected"]

    async def test_safety_complaint_raises_alert(self, make_executor):
        ex = make_executor("...")
        await ex.run("update_case", args(scenario="driver_complaint", reason_category="safety_threat"))
        assert ex.ctx.state.safety_alert

    async def test_bad_scenario(self, make_executor):
        result = await make_executor("...").run("update_case", args(scenario="refund"))
        assert result["ok"] is False


class TestPrice:
    async def test_price_comes_from_state_not_from_arguments(self, make_executor):
        ex = make_executor("...")
        missing = await ex.run("get_trip_price", args(origin="A", destination="B"))
        assert missing["ok"] is False  # no case yet: arguments are not used
        await ex.run("update_case", args(scenario="trip_price", origin="Крещатик 22", destination="Вокзал"))
        quote = await ex.run("get_trip_price", "{}")
        expected = await StubPriceProvider("UAH").quote("Крещатик 22", "Вокзал", "economy")
        assert quote["quote"]["price"] == expected.price
        assert ex.ctx.state.active_case.status == CaseStatus.DONE

    async def test_changing_an_address_drops_the_old_quote(self, make_executor):
        ex = make_executor("...")
        await ex.run("update_case", args(scenario="trip_price", origin="Крещатик 22", destination="Вокзал"))
        await ex.run("get_trip_price", "{}")
        await ex.run("update_case", args(scenario="trip_price", destination="Аэропорт Борисполь"))
        case = ex.ctx.state.active_case
        assert case.quote is None and case.status == CaseStatus.COLLECTING

    async def test_outage_returns_no_price_and_counts_failure(self, make_executor):
        ex = make_executor("...", pricing=UnavailablePriceProvider())
        await ex.run("update_case", args(scenario="trip_price", origin="Крещатик 22", destination="Вокзал"))
        result = await ex.run("get_trip_price", "{}")
        assert result["ok"] is False and "price" not in result
        assert ex.ctx.state.consecutive_failures == 1


class TestTicket:
    async def _ready_lost_item(self, ex, item="чёрный зонт"):
        await ex.run("update_case", args(scenario="lost_item", order_id="731842", item_description=item))

    async def test_blocked_while_fields_are_missing(self, make_executor):
        ex = make_executor("заказ 731842")
        await ex.run("update_case", args(scenario="lost_item", order_id="731842"))
        result = await ex.run("create_operator_ticket", args(summary="x"))
        assert result["ok"] is False and "item_description" in result["error"]

    async def test_blocked_before_trip_lookup(self, make_executor):
        ex = make_executor("заказ 731842")
        await self._ready_lost_item(ex)
        result = await ex.run("create_operator_ticket", args(summary="x"))
        assert result == {"ok": False, "error": "call find_trip before creating the ticket"}

    async def test_ticket_fields_come_from_state(self, make_executor):
        ex = make_executor("заказ 731842")
        await self._ready_lost_item(ex)
        await ex.run("find_trip", "{}")
        result = await ex.run("create_operator_ticket", args(summary="Клиент забыл зонт"))
        assert result["ok"] and result["ticket_id"].startswith("OP-")
        assert result["policy"]  # what-happens-next text comes from the knowledge base
        [ticket] = await ex.ctx.tickets.list("s1")
        assert ticket.fields["order_id"] == "731842"
        assert ticket.fields["found_trip"]["driver_name"] == "Олег"
        assert ticket.priority == "normal"
        assert ex.ctx.state.active_case.status == CaseStatus.DONE

    async def test_documents_get_high_priority(self, make_executor):
        ex = make_executor("заказ 731842")
        await self._ready_lost_item(ex, item="паспорт в синей обложке")
        await ex.run("find_trip", "{}")
        result = await ex.run("create_operator_ticket", args(summary="паспорт"))
        assert result["priority"] == "high"

    async def test_unmatched_trip_is_flagged_for_operator(self, make_executor):
        ex = make_executor("...")
        await ex.run(
            "update_case",
            args(scenario="lost_item", trip_date="2026-10-25", approx_time="03:00", item_description="рюкзак"),
        )
        await ex.run("find_trip", "{}")
        await ex.run("create_operator_ticket", args(summary="рюкзак"))
        [ticket] = await ex.ctx.tickets.list("s1")
        assert "search manually" in ticket.fields["note"]


async def test_handoff_freezes_session_and_keeps_case_data(make_executor):
    ex = make_executor("заказ 731655, водитель был пьяный")
    await ex.run("update_case", args(scenario=Scenario.DRIVER_COMPLAINT.value, order_id="731655"))
    result = await ex.run("handoff_to_operator", args(reason="пьяный водитель", priority="urgent"))
    state = ex.ctx.state
    assert state.status == "handed_off" and state.handoff.priority == "urgent"
    [ticket] = await ex.ctx.tickets.list("s1")
    assert ticket.id == result["ticket_id"] and ticket.fields["case"]["order_id"] == "731655"


async def test_kb_search_without_hits_tells_model_not_to_improvise(make_executor):
    result = await make_executor("...").run("search_knowledge_base", args(query="квантовая хромодинамика"))
    assert result["results"] == [] and "Do not answer from general knowledge" in result["note"]


async def test_unknown_tool_and_bad_json(make_executor):
    ex = make_executor("...")
    assert (await ex.run("delete_everything", "{}"))["ok"] is False
    assert (await ex.run("update_case", "{not json"))["ok"] is False


class TestLookupStability:
    async def test_copying_found_trip_data_keeps_the_lookup(self, make_executor):
        ex = make_executor("заказ 729904, забыла паспорт")
        await ex.run("update_case", args(scenario="lost_item", order_id="729904", item_description="паспорт"))
        await ex.run("find_trip", "{}")
        # models like to copy the found trip into the case; that must not force a second lookup
        result = await ex.run(
            "update_case",
            args(scenario="lost_item", trip_date="2026-10-24", approx_time="22:30", route="ЖК → центр"),
        )
        case = ex.ctx.state.active_case
        assert case.trip_lookup_done and case.found_trip and case.trip_confirmed
        assert result["next_step"] == "call create_operator_ticket"

    async def test_new_date_after_search_invalidates_the_lookup(self, make_executor):
        ex = make_executor("вчера около 20:00")
        await ex.run(
            "update_case",
            args(scenario="lost_item", trip_date="2026-10-25", approx_time="19:40", route="Крещатик вокзал"),
        )
        await ex.run("find_trip", "{}")
        assert ex.ctx.state.active_case.found_trip
        await ex.run("update_case", args(scenario="lost_item", approx_time="08:15"))
        assert not ex.ctx.state.active_case.trip_lookup_done


async def test_note_is_appended_to_existing_ticket(make_executor):
    ex = make_executor("заказ 731842")
    no_ticket = await ex.run("add_ticket_note", args(note="наклейка"))
    assert no_ticket["ok"] is False
    await ex.run("update_case", args(scenario="lost_item", order_id="731842", item_description="рюкзак"))
    await ex.run("find_trip", "{}")
    created = await ex.run("create_operator_ticket", args(summary="рюкзак"))
    result = await ex.run("add_ticket_note", args(note="На рюкзаке наклейка с котом"))
    assert result["ticket_id"] == created["ticket_id"]
    [ticket] = await ex.ctx.tickets.list("s1")
    assert ticket.notes == ["На рюкзаке наклейка с котом"]
