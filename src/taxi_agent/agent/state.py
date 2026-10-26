"""Dialog state owned by code, not by the model.

The LLM only proposes values through tools; this module decides what is required,
what is still missing and when a case is complete.
"""

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from taxi_agent.integrations.orders import Trip
from taxi_agent.integrations.pricing import PriceQuote


class Scenario(StrEnum):
    LOST_ITEM = "lost_item"
    TRIP_PRICE = "trip_price"
    DRIVER_COMPLAINT = "driver_complaint"


class CaseStatus(StrEnum):
    COLLECTING = "collecting"
    DONE = "done"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    HANDED_OFF = "handed_off"


class Tariff(StrEnum):
    ECONOMY = "economy"
    COMFORT = "comfort"
    BUSINESS = "business"


class ComplaintReason(StrEnum):
    RUDE_BEHAVIOR = "rude_behavior"
    DANGEROUS_DRIVING = "dangerous_driving"
    ROUTE_OR_FARE = "route_or_fare"
    CAR_CONDITION = "car_condition"
    LATE_OR_NO_SHOW = "late_or_no_show"
    SAFETY_THREAT = "safety_threat"
    OTHER = "other"


class Priority(StrEnum):
    NORMAL = "normal"
    HIGH = "high"  # e.g. forgotten documents, money, cards
    URGENT = "urgent"  # safety incidents


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


class _CaseBase(BaseModel):
    id: str = Field(default_factory=_new_id)
    status: CaseStatus = CaseStatus.COLLECTING
    ticket_id: str | None = None

    def missing_fields(self) -> list[str]:
        raise NotImplementedError


class _TripReference(BaseModel):
    """How a past trip is identified: an order number, or date + time (+ route)."""

    order_id: str | None = None
    trip_date: date | None = None
    approx_time: str | None = None  # HH:MM
    route: str | None = None
    found_trip: Trip | None = None  # set only by find_trip, never by the model
    candidate_order_ids: list[str] = Field(default_factory=list)  # all matches of the last find_trip
    trip_lookup_done: bool = False
    trip_confirmed: bool = False

    def _trip_missing(self) -> list[str]:
        if self.order_id:
            return []
        missing = []
        if not self.trip_date:
            missing.append("order_id_or_trip_date")
        if not self.approx_time:
            missing.append("approx_time" if self.trip_date else "order_id_or_approx_time")
        return missing


class LostItemCase(_CaseBase, _TripReference):
    scenario: Literal[Scenario.LOST_ITEM] = Scenario.LOST_ITEM
    item_description: str | None = None
    contact_phone: str | None = None

    def missing_fields(self) -> list[str]:
        missing = self._trip_missing()
        if not self.item_description:
            missing.append("item_description")
        return missing


class TripPriceCase(_CaseBase):
    scenario: Literal[Scenario.TRIP_PRICE] = Scenario.TRIP_PRICE
    origin: str | None = None
    destination: str | None = None
    tariff: Tariff = Tariff.ECONOMY
    quote: PriceQuote | None = None  # set only by get_trip_price

    def missing_fields(self) -> list[str]:
        return [name for name in ("origin", "destination") if not getattr(self, name)]


class DriverComplaintCase(_CaseBase, _TripReference):
    scenario: Literal[Scenario.DRIVER_COMPLAINT] = Scenario.DRIVER_COMPLAINT
    reason_category: ComplaintReason | None = None
    description: str | None = None
    safety_incident: bool = False

    def missing_fields(self) -> list[str]:
        missing = self._trip_missing()
        if not self.reason_category:
            missing.append("reason_category")
        if not self.description:
            missing.append("description")
        return missing


Case = Annotated[LostItemCase | TripPriceCase | DriverComplaintCase, Field(discriminator="scenario")]
CASE_TYPES: dict[Scenario, type[_CaseBase]] = {
    Scenario.LOST_ITEM: LostItemCase,
    Scenario.TRIP_PRICE: TripPriceCase,
    Scenario.DRIVER_COMPLAINT: DriverComplaintCase,
}


class Handoff(BaseModel):
    reason: str
    priority: Priority
    ticket_id: str
    at: datetime


class SessionState(BaseModel):
    session_id: str
    created_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE
    language: str = "ru"
    cases: list[Case] = Field(default_factory=list)
    active_case_id: str | None = None
    handoff: Handoff | None = None
    ticket_ids: list[str] = Field(default_factory=list)
    urgent_ticket_id: str | None = None  # one safety incident gets one urgent ticket

    # guardrail counters
    user_turns: int = 0
    stalled_turns: int = 0
    consecutive_failures: int = 0
    safety_alert: bool = False

    # token accounting for the demo UI
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def active_case(self) -> LostItemCase | TripPriceCase | DriverComplaintCase | None:
        return next((c for c in self.cases if c.id == self.active_case_id), None)

    def open_case(self, scenario: Scenario) -> LostItemCase | TripPriceCase | DriverComplaintCase:
        """Return the case to write into for this scenario, creating one if needed.

        A trip-price case stays editable after a quote (the customer may change an address);
        lost-item and complaint cases are frozen once their ticket exists, so new facts start a new case.
        """
        for case in reversed(self.cases):
            if case.scenario != scenario:
                continue
            if case.status == CaseStatus.COLLECTING or scenario == Scenario.TRIP_PRICE:
                self.active_case_id = case.id
                return case
            break
        case = CASE_TYPES[scenario]()
        self.cases.append(case)
        self.active_case_id = case.id
        return case

    def missing_snapshot(self) -> tuple[str | None, tuple[str, ...]]:
        case = self.active_case
        if case is None or case.status != CaseStatus.COLLECTING:
            return None, ()
        return case.id, tuple(case.missing_fields())
