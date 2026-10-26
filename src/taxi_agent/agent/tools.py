"""Tools exposed to the model.

Design rule: tools that act (find_trip, get_trip_price, create_operator_ticket) take their inputs from
the case state, not from model arguments. The model can only put data into the state via update_case,
where every value is validated and order numbers are checked against what the customer actually typed.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
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

logger = logging.getLogger(__name__)


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
        "add_ticket_note",
        "Add details the customer gives AFTER a ticket was created (a better description of the item, a "
        "contact number, what else happened) to that existing ticket. Use this instead of update_case when "
        "the case already has a ticket; only a different item or incident starts a new case.",
        {
            "note": {"type": "string", "description": "The new details, in Russian, for the operator."},
            "ticket_id": {
                "type": "string",
                "description": "Only when the details are for an earlier ticket of this chat, not the active case.",
            },
        },
        ["note"],
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

TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)
# only fields declared in the update_case schema; internal ones (found_trip, quote, flags) are set by code
_WRITABLE = frozenset(TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]) - {"scenario"}
_TRIP_REF_FIELDS = {"order_id", "trip_date", "approx_time", "route"}
_QUOTE_FIELDS = {"origin", "destination", "tariff"}
_TEXT_LIMIT = 500
_TIME = re.compile(r"^([01]?\d|2[0-3])[:.]([0-5]\d)$")
# Whole words only: "справа" is not "права" (driving licence), "картонная" is not "карта".
_VALUABLES = re.compile(
    r"\b(?:паспорт\w*|документ\w*|прав(?:а|ами)|водительск\w*|удостоверени\w*|посвідченн\w*|"
    r"кошел[её]к\w*|кошельк\w*|бумажник\w*|гаманц\w*|гаманець|портмоне|деньг\w*|налич\w*|грош\w*|"
    r"карт(?:а|у|ой|ы|очк\w*|ка|ку|кой)|кредитк\w*|банковск\w*|"
    r"телефон\w*|смартфон\w*|айфон\w*|iphone\w*|ноутбук\w*|макбук\w*|macbook\w*|laptop\w*|планшет\w*|ipad\w*|"
    r"наушник\w*|навушник\w*|airpods\w*|ключ(?:и|ами|ей)?|ключі|wallet\w*|passport\w*|money|cash|cards?|"
    r"phones?|keys?)\b",
    re.IGNORECASE,
)
_FUTURE_TOLERANCE = timedelta(minutes=15)

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
    turn_failed: bool = False  # an external service failed during this turn (counted once per turn)


def _trip_view(trip: Trip) -> dict[str, Any]:
    return trip.model_dump(mode="json")


def _case_view(case: LostItemCase | TripPriceCase | DriverComplaintCase) -> dict[str, Any]:
    return case.model_dump(mode="json", exclude={"id"})


def _lookup_invalidated(case: LostItemCase | DriverComplaintCase, applied: set[str]) -> bool:
    """Does this update make the previous find_trip result stale?

    A lookup by order number does not depend on date/time/route, and re-saving the found trip's own
    date and time (models like to copy them into the case) changes nothing either.
    """
    changed = _TRIP_REF_FIELDS & applied
    if not changed:
        return False
    if "order_id" in changed:
        return not (case.found_trip and case.found_trip.order_id == case.order_id)
    if case.order_id:
        return False
    trip = case.found_trip
    if trip is None:
        return True
    same_date = "trip_date" not in changed or case.trip_date == trip.date
    same_time = "approx_time" not in changed or case.approx_time == trip.time
    return not (same_date and same_time)


def _next_step(case: LostItemCase | TripPriceCase | DriverComplaintCase) -> str:
    missing = case.missing_fields()
    if isinstance(case, TripPriceCase):
        if missing:
            return f"ask the customer for: {', '.join(missing)}"
        if case.quote is None:
            return "call get_trip_price"
        return "tell the customer the quoted price"
    if case.status == CaseStatus.DONE:
        return f"ticket {case.ticket_id} already exists; put any extra details into add_ticket_note"
    if not case.trip_lookup_done and (case.order_id or (case.trip_date and case.approx_time)):
        if missing:
            return f"call find_trip, then ask for: {', '.join(missing)}"
        return "call find_trip"
    if case.trip_lookup_done and not case.found_trip and len(case.candidate_order_ids) > 1:
        return (
            "ask the customer which of the matching trips is theirs (route or time), then save their choice "
            "with update_case(order_id=<order id of that trip>)"
        )
    if case.found_trip and not case.trip_confirmed:
        return (
            "show the found trip (date, time, route) and ask ONLY whether it is the customer's trip — no other "
            "question in this message; other details come after the confirmation"
        )
    if missing:
        return f"ask the customer for: {', '.join(missing)}"
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
        if name not in TOOL_NAMES:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        handler = getattr(self, f"_{name}")
        try:
            return await handler(args)
        except Exception:  # a failing integration (orders, pricing, tickets) must not crash the turn
            logger.exception("tool %s failed", name)
            self.ctx.turn_failed = True
            return {
                "ok": False,
                "error": "the service behind this tool is temporarily unavailable",
                "next_step": "apologize, say you can't check this right now and offer to try again later or "
                "to pass the request to an operator. Do not invent any data.",
            }

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

        previous = {name: getattr(case, name, None) for name in ("trip_date", "approx_time")}
        for name, raw in args.items():
            if raw is None or raw == "":
                continue
            if name not in _WRITABLE or name not in type(case).model_fields:
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
            applied[name] = value.isoformat() if isinstance(value, date) else value

        if not isinstance(case, TripPriceCase) and {"trip_date", "approx_time"} & applied.keys():
            # each value can be valid alone while "today at 23:00" at 14:00 is not
            now = self.ctx.now.replace(tzinfo=None)
            if case.trip_date and case.approx_time:
                moment = datetime.combine(case.trip_date, time.fromisoformat(case.approx_time))
                if moment > now + _FUTURE_TOLERANCE:
                    for name in ("trip_date", "approx_time"):
                        if name in applied:
                            setattr(case, name, previous[name])
                            applied.pop(name)
                            rejected[name] = "this date and time are still in the future; ask the customer to check"
        if not isinstance(case, TripPriceCase) and _lookup_invalidated(case, set(applied)):
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
            if guardrails.looks_like_phone(digits):
                raise ValueError("this looks like a phone number, not an order number; ask for the order number")
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
        if name == "contact_phone":
            digits = re.sub(r"\D", "", str(raw))
            if not guardrails.is_grounded_phone(digits, self.ctx.messages):
                raise ValueError("the customer did not write this phone number; ask them for it")
            return f"+{digits}" if digits.startswith("380") else digits
        if name in {"safety_incident", "trip_confirmed"}:
            if not isinstance(raw, bool):
                raise ValueError("must be true or false")
            if name == "trip_confirmed" and raw and not getattr(case, "found_trip", None):
                raise ValueError("there is no found trip to confirm; call find_trip first")
            return raw
        # one line: customer text is rendered into the system prompt and must not fake extra lines there
        text = " ".join(str(raw).split())
        if len(text) > _TEXT_LIMIT:
            text = text[:_TEXT_LIMIT]
        return text

    # --- find_trip -------------------------------------------------------------------------------
    async def _find_trip(self, args: dict[str, Any]) -> dict[str, Any]:
        case = self.ctx.state.active_case
        if not isinstance(case, LostItemCase | DriverComplaintCase):
            return {"ok": False, "error": "no active lost-item or complaint case; call update_case first"}
        by_order = bool(case.order_id)
        wrong_order_id = None
        if not by_order and not (case.trip_date and case.approx_time):
            return {
                "ok": False,
                "error": "save the order number or the trip date and time with update_case first",
            }
        matches = []
        if by_order:
            trip = await self.ctx.orders.get(case.order_id)
            if trip:
                matches = [trip]
            else:
                # a wrong number must not block the search by date and time
                wrong_order_id, case.order_id, by_order = case.order_id, None, False
        if not by_order and case.trip_date and case.approx_time:
            matches = await self.ctx.orders.search(case.trip_date, case.approx_time, case.route)

        case.trip_lookup_done = True
        case.found_trip = matches[0] if len(matches) == 1 else None
        case.candidate_order_ids = [t.order_id for t in matches]
        case.trip_confirmed = bool(case.found_trip and by_order)

        if not matches:
            note = (
                "No trip found. Tell the customer; if they are sure about the details, the ticket can still be "
                "created and the operator will search manually."
            )
            if wrong_order_id:
                note = (
                    f"No trip with order number {wrong_order_id}; it was removed from the case. Ask the customer "
                    "to check the number or to give the trip date and approximate time."
                )
        elif len(matches) == 1:
            note = (
                "Trip found by order number and linked to the case; do not ask the customer to confirm it."
                if by_order
                else "One trip matches. Ask ONLY whether it is the customer's trip before anything else."
            )
        else:
            note = "Several trips match. Ask the customer which one (route or exact time); do not guess."
        if wrong_order_id and matches:
            note = f"Order number {wrong_order_id} does not exist; searched by date and time instead. " + note
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
                self.ctx.turn_failed = True
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
        if case.found_trip and not case.trip_confirmed:
            return {"ok": False, "error": "confirm the found trip with the customer first (date, time, route)"}
        if not case.found_trip and len(case.candidate_order_ids) > 1:
            return {"ok": False, "error": "several trips match; ask the customer which one is theirs first"}

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
        if priority == Priority.URGENT:
            state.urgent_ticket_id = ticket.id
        policy = [
            f"{section}: {chunk.text}"
            for source, section in TICKET_POLICY[case.scenario]
            for chunk in self.ctx.kb.chunks
            if chunk.source == source and chunk.section == section
        ]
        return {"ok": True, "ticket_id": ticket.id, "priority": priority.value, "policy": policy}

    # --- add_ticket_note -------------------------------------------------------------------------
    async def _add_ticket_note(self, args: dict[str, Any]) -> dict[str, Any]:
        note = str(args.get("note", "")).strip()[:_TEXT_LIMIT]
        if not note:
            return {"ok": False, "error": "note is empty"}
        state = self.ctx.state
        requested = str(args.get("ticket_id") or "").strip()
        if requested and requested not in state.ticket_ids:
            return {"ok": False, "error": f"there is no ticket {requested} in this chat"}
        # by default only the active case's ticket: a new complaint's details must not land in the lost-item ticket
        active = state.active_case
        ticket_id = requested or (active.ticket_id if active else None)
        if ticket_id is None:
            return {
                "ok": False,
                "error": "the active case has no ticket yet; collect its data and call create_operator_ticket",
            }
        await self.ctx.tickets.add_note(ticket_id, note)
        return {"ok": True, "ticket_id": ticket_id, "next_step": "tell the customer the details were added"}

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
    if state.safety_alert:
        priority = Priority.URGENT  # whoever triggers the handoff, a safety incident is always urgent
        if state.urgent_ticket_id:  # one incident — one urgent ticket (e.g. an urgent complaint already exists)
            state.status = SessionStatus.HANDED_OFF
            state.handoff = Handoff(reason=reason, priority=priority, ticket_id=state.urgent_ticket_id, at=ctx.now)
            return state.urgent_ticket_id
    case = state.active_case
    ticket_id = await _create_handoff_ticket(ctx, reason, priority)
    state.status = SessionStatus.HANDED_OFF
    state.handoff = Handoff(reason=reason, priority=priority, ticket_id=ticket_id, at=ctx.now)
    if case is not None and case.status == CaseStatus.COLLECTING:
        case.ticket_id = ticket_id
    return ticket_id


async def escalate_to_urgent(ctx: ToolContext, reason: str) -> str:
    """A safety report in a session that is already with an operator at normal priority:
    open an urgent ticket so the on-duty operator sees it (at night only urgent ones are handled)."""
    state = ctx.state
    if state.urgent_ticket_id:
        return state.urgent_ticket_id
    ticket_id = await _create_handoff_ticket(ctx, reason, Priority.URGENT)
    previous = state.handoff.ticket_id if state.handoff else None
    state.handoff = Handoff(reason=reason, priority=Priority.URGENT, ticket_id=ticket_id, at=ctx.now)
    if previous:
        await ctx.tickets.add_note(previous, f"Эскалировано: срочное обращение {ticket_id}")
    return ticket_id


async def _create_handoff_ticket(ctx: ToolContext, reason: str, priority: Priority) -> str:
    state = ctx.state
    case = state.active_case
    fields: dict[str, Any] = {"reason": reason}
    if case is not None:
        fields["case"] = case.model_dump(mode="json", exclude={"id"})
    if state.handoff:
        fields["previous_ticket"] = state.handoff.ticket_id
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
    state.ticket_ids.append(ticket.id)
    if priority == Priority.URGENT:
        state.urgent_ticket_id = ticket.id
    return ticket.id
