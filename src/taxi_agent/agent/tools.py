"""Tools exposed to the model.

Design rule: tools that act (find_trip, get_trip_price, create_operator_ticket) take their inputs from
the case state, not from model arguments. The model can only put data into the state via update_case,
where every value is validated and order numbers are checked against what the customer actually typed.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from taxi_agent.agent import guardrails
from taxi_agent.agent.state import (
    CaseStatus,
    ComplaintReason,
    DriverComplaintCase,
    Handoff,
    LostItemCase,
    Priority,
    Scenario,
    SessionState,
    SessionStatus,
    Tariff,
    TripPriceCase,
)
from taxi_agent.integrations.orders import OrderRepository, Trip
from taxi_agent.integrations.pricing import PriceProvider, PricingError
from taxi_agent.integrations.tickets import TicketDraft, TicketSink
from taxi_agent.kb import KnowledgeBase


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn(
        "update_case",
        "Save facts the customer has just stated into the case for the given scenario (opens the case if "
        "needed). Pass only fields the customer actually provided in the conversation; omit everything "
        "else. Returns the case, rejected values and what is still missing.",
        {
            "scenario": {"type": "string", "enum": [s.value for s in Scenario]},
            "order_id": {"type": "string", "description": "Order number exactly as the customer gave it."},
            "trip_date": {"type": "string", "description": "Trip date, YYYY-MM-DD."},
            "approx_time": {"type": "string", "description": "Approximate trip start time, HH:MM (24h)."},
            "route": {
                "type": "string",
                "description": "Where the trip went from/to, in the customer's words.",
            },
            "item_description": {
                "type": "string",
                "description": "Lost item: what it is, color, distinctive details, where in the car.",
            },
            "contact_phone": {
                "type": "string",
                "description": "Only if the customer offers a contact number.",
            },
            "origin": {"type": "string", "description": "Pickup address for a price quote."},
            "destination": {"type": "string", "description": "Drop-off address for a price quote."},
            "tariff": {"type": "string", "enum": [t.value for t in Tariff]},
            "reason_category": {"type": "string", "enum": [r.value for r in ComplaintReason]},
            "description": {"type": "string", "description": "Complaint: what happened, in a few sentences."},
            "safety_incident": {"type": "boolean", "description": "True if anyone's safety was at risk."},
            "trip_confirmed": {
                "type": "boolean",
                "description": "True only after the customer confirmed that the found trip is the right one.",
            },
        },
        ["scenario"],
    ),
    _fn(
        "find_trip",
        "Look up the customer's past trip for the active lost-item or complaint case using the order number, "
        "or the date + approximate time (+ route) already saved in the case. Takes no arguments.",
        {},
        [],
    ),
    _fn(
        "get_trip_price",
        "Get a price quote from the pricing service for the active trip-price case, using the origin, "
        "destination and tariff saved in the case. Takes no arguments.",
        {},
        [],
    ),
    _fn(
        "search_knowledge_base",
        "Search the company's help articles (lost-and-found rules, tariffs, payment, waiting, cancellation, "
        "complaints process, support hours). Write the query in Russian — the articles are in Russian.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _fn(
        "create_operator_ticket",
        "Create the ticket for a human operator for the active lost-item or complaint case. Structured data "
        "is taken from the case automatically; you only write a short summary.",
        {
            "summary": {
                "type": "string",
                "description": "1–3 sentences in Russian for the operator: what happened and what the customer wants.",
            }
        },
        ["summary"],
    ),
    _fn(
        "handoff_to_operator",
        "Pass the whole conversation to a human operator right now. After this the agent stops answering.",
        {
            "reason": {"type": "string", "description": "Short reason, in Russian."},
            "priority": {"type": "string", "enum": [p.value for p in Priority]},
        },
        ["reason", "priority"],
    ),
]

_TRIP_REF_FIELDS = {"order_id", "trip_date", "approx_time", "route"}
_QUOTE_FIELDS = {"origin", "destination", "tariff"}
_TEXT_LIMIT = 500
_TIME = re.compile(r"^([01]?\d|2[0-3])[:.]([0-5]\d)$")
_VALUABLES = re.compile(
    r"паспорт|документ|права|кошел|гаманец|деньг|грош|карт|card|wallet|passport|money|ноутбук|laptop|"
    r"телефон|phone|ключ|key",
    re.IGNORECASE,
)

TICKET_POLICY = {
    Scenario.LOST_ITEM: [("lost_items.md", "Сроки")],
    Scenario.DRIVER_COMPLAINT: [("complaints.md", "Как рассматривается жалоба")],
}


@dataclass
class ToolContext:
    state: SessionState
    messages: list[dict[str, Any]]  # full history including the current user message
    now: datetime
    orders: OrderRepository
    pricing: PriceProvider
    tickets: TicketSink
    kb: KnowledgeBase
    calls: list[str] = field(default_factory=list)


def _trip_view(trip: Trip) -> dict[str, Any]:
    return trip.model_dump(mode="json")


def _case_view(case: LostItemCase | TripPriceCase | DriverComplaintCase) -> dict[str, Any]:
    return case.model_dump(mode="json", exclude={"id"})


def _next_step(case: LostItemCase | TripPriceCase | DriverComplaintCase) -> str:
    missing = case.missing_fields()
    if isinstance(case, TripPriceCase):
        if missing:
            return f"ask the customer for: {', '.join(missing)}"
        if case.quote is None:
            return "call get_trip_price"
        return "tell the customer the quoted price"
    if case.status == CaseStatus.DONE:
        return f"ticket {case.ticket_id} already exists for this case"
    if not case.trip_lookup_done and (case.order_id or (case.trip_date and case.approx_time)):
        if missing:
            return f"call find_trip, then ask for: {', '.join(missing)}"
        return "call find_trip"
    if case.trip_lookup_done and not case.found_trip and len(case.candidate_order_ids) > 1:
        return (
            "ask the customer which of the matching trips is theirs (route or time), then save their choice "
            "with update_case(order_id=<order id of that trip>)"
        )
    if missing:
        return f"ask the customer for: {', '.join(missing)}"
    if case.found_trip and not case.trip_confirmed:
        return "confirm the found trip with the customer (date, time, route), then create_operator_ticket"
    return "call create_operator_ticket"


class ToolExecutor:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    async def run(self, name: str, raw_arguments: str) -> dict[str, Any]:
        self.ctx.calls.append(name)
        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            return {"ok": False, "error": f"invalid JSON arguments: {exc}"}
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        return await handler(args)

    # --- update_case -----------------------------------------------------------------------------
    async def _update_case(self, args: dict[str, Any]) -> dict[str, Any]:
        state = self.ctx.state
        try:
            scenario = Scenario(args.pop("scenario", None))
        except ValueError:
            return {"ok": False, "error": f"scenario must be one of {[s.value for s in Scenario]}"}
        case = state.open_case(scenario)
        applied: dict[str, Any] = {}
        rejected: dict[str, str] = {}
        ignored: list[str] = []

        for name, raw in args.items():
            if raw is None or raw == "":
                continue
            if name not in type(case).model_fields or name in {"id", "status", "ticket_id", "scenario"}:
                ignored.append(name)
                continue
            try:
                value = self._validate(name, raw, case)
            except ValueError as exc:
                rejected[name] = str(exc)
                continue
            if getattr(case, name) == value:
                continue
            setattr(case, name, value)
            applied[name] = value

        if not isinstance(case, TripPriceCase) and _TRIP_REF_FIELDS & applied.keys():
            case.found_trip, case.trip_lookup_done, case.trip_confirmed = None, False, False
        if isinstance(case, TripPriceCase) and _QUOTE_FIELDS & applied.keys():
            case.quote, case.status = None, CaseStatus.COLLECTING
        if isinstance(case, DriverComplaintCase) and (
            case.safety_incident or case.reason_category == ComplaintReason.SAFETY_THREAT
        ):
            state.safety_alert = True

        result: dict[str, Any] = {
            "ok": True,
            "case": _case_view(case),
            "applied": applied,
            "missing_fields": case.missing_fields(),
            "next_step": _next_step(case),
        }
        if rejected:
            result["rejected"] = rejected
        if ignored:
            result["ignored_fields"] = ignored
        return result

    def _validate(self, name: str, raw: Any, case: Any) -> Any:
        today = self.ctx.now.date()
        if name == "order_id":
            digits = re.sub(r"\D", "", str(raw))
            if not 5 <= len(digits) <= 10:
                raise ValueError("an order number has 5–10 digits; ask the customer to check it")
            from_lookup = digits in getattr(case, "candidate_order_ids", [])
            if not from_lookup and not guardrails.is_grounded_order_id(digits, self.ctx.messages):
                raise ValueError("neither the customer nor find_trip gave this number; ask the customer for it")
            return digits
        if name == "trip_date":
            try:
                value = date.fromisoformat(str(raw))
            except ValueError:
                raise ValueError("use YYYY-MM-DD; if the customer was vague, ask which day") from None
            if value > today:
                raise ValueError("the date is in the future; ask the customer to check it")
            if value < today - timedelta(days=90):
                raise ValueError("trips older than 90 days are not handled in chat; offer an operator")
            return value
        if name == "approx_time":
            match = _TIME.match(str(raw).strip())
            if not match:
                raise ValueError("use HH:MM; if the customer gave only a part of day, ask for an approximate hour")
            return f"{int(match.group(1)):02d}:{match.group(2)}"
        if name == "tariff":
            return Tariff(str(raw).lower())
        if name == "reason_category":
            return ComplaintReason(str(raw).lower())
        if name in {"safety_incident", "trip_confirmed"}:
            if not isinstance(raw, bool):
                raise ValueError("must be true or false")
            if name == "trip_confirmed" and raw and not getattr(case, "found_trip", None):
                raise ValueError("there is no found trip to confirm; call find_trip first")
            return raw
        text = str(raw).strip()
        if len(text) > _TEXT_LIMIT:
            text = text[:_TEXT_LIMIT]
        return text

    # --- find_trip -------------------------------------------------------------------------------
    async def _find_trip(self, args: dict[str, Any]) -> dict[str, Any]:
        case = self.ctx.state.active_case
        if not isinstance(case, LostItemCase | DriverComplaintCase):
            return {"ok": False, "error": "no active lost-item or complaint case; call update_case first"}
        by_order = bool(case.order_id)
        if by_order:
            trip = await self.ctx.orders.get(case.order_id)
            matches = [trip] if trip else []
        elif case.trip_date and case.approx_time:
            matches = await self.ctx.orders.search(case.trip_date, case.approx_time, case.route)
        else:
            return {
                "ok": False,
                "error": "save the order number or the trip date and time with update_case first",
            }

        case.trip_lookup_done = True
        case.found_trip = matches[0] if len(matches) == 1 else None
        case.candidate_order_ids = [t.order_id for t in matches]
        case.trip_confirmed = bool(case.found_trip and by_order)

        if not matches:
            note = (
                "No trip found. Tell the customer; if they are sure about the details, the ticket can still be "
                "created and the operator will search manually."
            )
            if by_order:
                note = "No trip with this order number. Ask the customer to check the number or give date and time."
        elif len(matches) == 1:
            note = (
                "Trip found by order number."
                if by_order
                else "One trip matches. Confirm it with the customer (date, time, route) before the ticket."
            )
        else:
            note = "Several trips match. Ask the customer which one (route or exact time); do not guess."
        return {
            "ok": True,
            "matches": [_trip_view(t) for t in matches[:5]],
            "note": note,
            "next_step": _next_step(case),
        }

    # --- get_trip_price --------------------------------------------------------------------------
    async def _get_trip_price(self, args: dict[str, Any]) -> dict[str, Any]:
        state = self.ctx.state
        case = state.active_case
        if not isinstance(case, TripPriceCase):
            return {
                "ok": False,
                "error": "no active trip-price case; call update_case with scenario trip_price",
            }
        if case.missing_fields():
            return {"ok": False, "error": f"missing: {', '.join(case.missing_fields())}; ask the customer"}
        try:
            quote = await self.ctx.pricing.quote(case.origin, case.destination, case.tariff.value)
        except PricingError as exc:
            if exc.retryable:
                state.consecutive_failures += 1
                return {
                    "ok": False,
                    "error": "pricing service is temporarily unavailable",
                    "next_step": "apologize, say the price can't be calculated right now; offer to try again "
                    "in a minute or to check it in the app. Do not name any price.",
                }
            return {
                "ok": False,
                "error": str(exc),
                "next_step": "ask the customer to clarify the addresses (street and house number).",
            }
        state.consecutive_failures = 0
        case.quote = quote
        case.status = CaseStatus.DONE
        return {
            "ok": True,
            "quote": quote.model_dump(mode="json", exclude={"valid_until"}),
            "valid_minutes": 10,
            "next_step": "tell the customer the price (preliminary estimate), tariff and approximate trip time",
        }

    # --- search_knowledge_base -------------------------------------------------------------------
    async def _search_knowledge_base(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is empty"}
        hits = self.ctx.kb.search(query, k=3)
        if not hits:
            return {
                "ok": True,
                "results": [],
                "note": "Nothing relevant found. Do not answer from general knowledge; offer an operator.",
            }
        return {
            "ok": True,
            "results": [
                {"article": c.title, "section": c.section, "text": c.text, "score": round(s, 2)} for c, s in hits
            ],
        }

    # --- create_operator_ticket ------------------------------------------------------------------
    async def _create_operator_ticket(self, args: dict[str, Any]) -> dict[str, Any]:
        state = self.ctx.state
        case = state.active_case
        if not isinstance(case, LostItemCase | DriverComplaintCase):
            return {"ok": False, "error": "tickets are created only for lost-item and complaint cases"}
        if case.ticket_id:
            return {"ok": True, "ticket_id": case.ticket_id, "note": "ticket already exists"}
        if missing := case.missing_fields():
            return {"ok": False, "error": f"cannot create the ticket, missing: {', '.join(missing)}"}
        if not case.trip_lookup_done:
            return {"ok": False, "error": "call find_trip before creating the ticket"}

        priority = Priority.NORMAL
        if isinstance(case, DriverComplaintCase) and (
            case.safety_incident or case.reason_category == ComplaintReason.SAFETY_THREAT
        ):
            priority = Priority.URGENT
        elif isinstance(case, LostItemCase) and _VALUABLES.search(case.item_description or ""):
            priority = Priority.HIGH

        fields = case.model_dump(mode="json", exclude={"id", "status", "ticket_id", "trip_lookup_done"})
        if case.found_trip is None:
            fields["note"] = "trip not matched automatically, operator should search manually"
        ticket = await self.ctx.tickets.create(
            TicketDraft(
                session_id=state.session_id,
                kind=case.scenario.value,
                priority=priority.value,
                summary=str(args.get("summary", "")).strip()[:_TEXT_LIMIT] or "(без описания)",
                fields=fields,
                customer_language=state.language,
                transcript=_visible_transcript(self.ctx.messages),
            )
        )
        case.ticket_id = ticket.id
        case.status = CaseStatus.DONE
        state.ticket_ids.append(ticket.id)
        policy = [
            f"{section}: {chunk.text}"
            for source, section in TICKET_POLICY[case.scenario]
            for chunk in self.ctx.kb.chunks
            if chunk.source == source and chunk.section == section
        ]
        return {"ok": True, "ticket_id": ticket.id, "priority": priority.value, "policy": policy}

    # --- handoff_to_operator ---------------------------------------------------------------------
    async def _handoff_to_operator(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            priority = Priority(str(args.get("priority", "normal")).lower())
        except ValueError:
            priority = Priority.NORMAL
        reason = str(args.get("reason", "")).strip()[:_TEXT_LIMIT] or "запрос клиента"
        ticket_id = await handoff(self.ctx, reason, priority)
        return {
            "ok": True,
            "ticket_id": ticket_id,
            "next_step": "tell the customer an operator will continue in this chat; if safety is at risk, "
            "mention 112. Do not ask further questions.",
        }


def _visible_transcript(messages: list[dict[str, Any]], limit: int = 16) -> list[dict[str, str]]:
    visible = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    return visible[-limit:]


async def handoff(ctx: ToolContext, reason: str, priority: Priority) -> str:
    """Create a handoff ticket with everything known so far and freeze the session."""
    state = ctx.state
    if state.handoff:
        return state.handoff.ticket_id
    case = state.active_case
    fields: dict[str, Any] = {"reason": reason}
    if case is not None:
        fields["case"] = case.model_dump(mode="json", exclude={"id"})
    ticket = await ctx.tickets.create(
        TicketDraft(
            session_id=state.session_id,
            kind="handoff",
            priority=priority.value,
            summary=reason,
            fields=fields,
            customer_language=state.language,
            transcript=_visible_transcript(ctx.messages),
        )
    )
    state.status = SessionStatus.HANDED_OFF
    state.handoff = Handoff(reason=reason, priority=priority, ticket_id=ticket.id, at=ctx.now)
    state.ticket_ids.append(ticket.id)
    if case is not None and case.status == CaseStatus.COLLECTING:
        case.ticket_id = ticket.id
    return ticket.id
