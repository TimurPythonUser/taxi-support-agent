"""State, stubs and knowledge base — no LLM involved."""

from datetime import date

import pytest

from taxi_agent.agent.state import (
    CaseStatus,
    DriverComplaintCase,
    LostItemCase,
    Scenario,
    SessionState,
    TripPriceCase,
)
from taxi_agent.integrations.orders import StubOrderRepository
from taxi_agent.integrations.pricing import PricingError, StubPriceProvider
from taxi_agent.kb import KnowledgeBase

from .helpers import KYIV, NOW


class TestMissingFields:
    def test_lost_item_needs_trip_reference_and_item(self):
        assert LostItemCase().missing_fields() == [
            "order_id_or_trip_date",
            "order_id_or_approx_time",
            "item_description",
        ]

    def test_order_id_alone_identifies_the_trip(self):
        assert LostItemCase(order_id="731842").missing_fields() == ["item_description"]

    def test_date_without_time_asks_for_time(self):
        case = LostItemCase(trip_date=date(2026, 10, 25), item_description="зонт")
        assert case.missing_fields() == ["approx_time"]

    def test_price_needs_both_addresses(self):
        assert TripPriceCase(origin="Крещатик 22").missing_fields() == ["destination"]

    def test_complaint_needs_reason_and_description(self):
        case = DriverComplaintCase(order_id="731655")
        assert case.missing_fields() == ["reason_category", "description"]


class TestOpenCase:
    def test_reuses_collecting_case(self):
        state = SessionState(session_id="s", created_at=NOW)
        first = state.open_case(Scenario.LOST_ITEM)
        assert state.open_case(Scenario.LOST_ITEM) is first

    def test_closed_lost_item_starts_new_case(self):
        state = SessionState(session_id="s", created_at=NOW)
        first = state.open_case(Scenario.LOST_ITEM)
        first.status = CaseStatus.DONE
        second = state.open_case(Scenario.LOST_ITEM)
        assert second is not first and state.active_case is second

    def test_price_case_stays_editable_after_quote(self):
        state = SessionState(session_id="s", created_at=NOW)
        price = state.open_case(Scenario.TRIP_PRICE)
        price.status = CaseStatus.DONE
        assert state.open_case(Scenario.TRIP_PRICE) is price

    def test_state_roundtrips_through_json(self):
        state = SessionState(session_id="s", created_at=NOW)
        state.open_case(Scenario.DRIVER_COMPLAINT).order_id = "731655"
        restored = SessionState.model_validate_json(state.model_dump_json())
        assert isinstance(restored.active_case, DriverComplaintCase)
        assert restored.active_case.order_id == "731655"


class TestPricingStub:
    async def test_same_route_same_price(self):
        pricing = StubPriceProvider("UAH")
        a = await pricing.quote("Крещатик 22", "Аэропорт Борисполь", "economy")
        b = await pricing.quote("крещатик, 22", "аэропорт борисполь", "economy")
        assert a.price == b.price and a.currency == "UAH"

    async def test_tariffs_are_ordered(self):
        pricing = StubPriceProvider("UAH")
        prices = [(await pricing.quote("Крещатик 22", "Оболонь", t)).price for t in ("economy", "comfort", "business")]
        assert prices == sorted(prices) and len(set(prices)) == 3

    async def test_min_fare(self):
        quote = await StubPriceProvider("UAH").quote("ул. Антоновича 50", "ул. Антоновича 52", "economy")
        assert quote.price >= StubPriceProvider.MIN_FARE

    @pytest.mark.parametrize(("origin", "destination"), [("Крещатик 22", "Крещатик 22"), ("12", "Оболонь")])
    async def test_bad_addresses_are_not_retryable(self, origin, destination):
        with pytest.raises(PricingError) as err:
            await StubPriceProvider("UAH").quote(origin, destination, "economy")
        assert not err.value.retryable


class TestOrdersStub:
    @pytest.fixture
    def orders(self):
        return StubOrderRepository(KYIV, "UAH", now=NOW)

    async def test_get_by_id(self, orders):
        trip = await orders.get("731842")
        assert trip and trip.date == date(2026, 10, 25) and trip.time == "19:40"

    async def test_two_trips_around_8pm_yesterday(self, orders):
        matches = await orders.search(date(2026, 10, 25), "20:00", None)
        assert [t.order_id for t in matches] == ["731842", "731870"]

    async def test_route_narrows_the_search(self, orders):
        matches = await orders.search(date(2026, 10, 25), "20:00", "с Крещатика на вокзал")
        assert [t.order_id for t in matches] == ["731842"]

    async def test_nothing_far_from_any_trip(self, orders):
        assert await orders.search(date(2026, 10, 25), "03:00", None) == []


@pytest.fixture(scope="module")
def kb():
    from taxi_agent.config import Settings

    return KnowledgeBase(Settings(_env_file=None).knowledge_base_dir)


class TestKnowledgeBase:
    @pytest.mark.parametrize(
        ("query", "section"),
        [
            ("сколько дней хранятся найденные вещи", "Срок хранения"),
            ("платное ожидание водителя", "Ожидание"),
            ("сколько стоит отменить заказ", "Отмена заказа"),
            ("как долго рассматривают жалобу на водителя", "Как рассматривается жалоба"),
            ("можно ли с собакой", "Детское кресло и животные"),
        ],
    )
    def test_finds_the_right_section(self, kb, query, section):
        hits = kb.search(query)
        assert hits and hits[0][0].section == section

    def test_unrelated_query_returns_nothing(self, kb):
        assert kb.search("квантовая хромодинамика глюоны") == []
