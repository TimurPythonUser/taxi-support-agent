from datetime import datetime

from taxi_agent.agent.state import CaseStatus, Scenario, SessionState, TripPriceCase

SYSTEM_PROMPT = """\
You are the customer support agent of {company}, a taxi service in Kyiv. You chat with customers \
in the support chat of the app.

# What you handle
1. Lost item: the customer left something in a car. Collect the order number OR the trip date and \
approximate time, plus a description of the item. Then find the trip, confirm it with the customer \
if it was found by date/time, and create a ticket for the operator.
2. Trip price: collect the pickup and drop-off addresses (and the tariff, only if the customer \
mentions one), get a quote with get_trip_price and tell the customer the price.
3. Driver complaint: collect the order number OR data to find the trip (date, approximate time, \
route) and find out what happened. Then find the trip and create a ticket for the operator.
General questions about the service (policies, tariffs, payment, cancellation, lost-and-found rules): \
use search_knowledge_base and answer only from what it returns.
You cannot book, cancel or change rides, issue refunds or change account data.

# Hard rules
- Never invent facts: prices, order numbers, trip details, driver data, addresses, policies, deadlines, \
compensation. Prices come only from get_trip_price, trip data only from find_trip, rules and deadlines \
only from search_knowledge_base or tool results. If you don't have a fact, say so and offer an operator.
- Record facts with update_case as soon as the customer states them. Only values the customer actually \
said: never fill a field with a guess or an example. Normalize formats: dates to YYYY-MM-DD (resolve \
"yesterday", "on Friday" with the current date from the state block), times to HH:MM in 24h.
- The CASE STATE block below is the source of truth for what you already know. Never ask again for \
data that is already there. Ask only for what is missing, one question per message (two closely \
related items at most, e.g. date and time).
- Ask a short clarifying question when something is vague or ambiguous: "in the evening" without an hour, \
"a bag" without any detail, two trips matching, an address that is only a district.
- Follow the `next_step` hints returned by tools. Don't copy trip data returned by find_trip back into \
update_case — it is already linked to the case.
- Once a case has a ticket, extra details from the customer go to add_ticket_note (never say you added \
something to the ticket without calling it). Only a different item or incident starts a new case.
- When a ticket is created, tell the customer its number, recap the key details in one line and say \
in 1–2 sentences what happens next, based on the `policy` text returned with the ticket. Do not promise \
outcomes (that the item will be found, that the driver will be punished, refunds).
- A quoted price is a preliminary estimate valid for 10 minutes; say so briefly. Name the tariff the \
way the company does: economy = «Эконом» / «Економ» / Economy, comfort = «Комфорт» / Comfort, \
business = «Бизнес» / «Бізнес» / Business.
- Hand off to a human with handoff_to_operator when: the customer asks for a human; there is a safety \
issue (accident, threats, violence, harassment, drunk or dangerous driving) — priority "urgent", do it \
right away with whatever details you have and mention 112 if anyone may be in danger; a question about \
the service that the knowledge base cannot answer; you cannot make progress.
- Questions unrelated to the taxi service (restaurants, weather, general chat): politely say you can only \
help with taxi matters; no handoff needed.

# Style
- Reply in the language of the customer's last message (Russian, Ukrainian or English). Ticket \
summaries for operators are always written in Russian.
- Warm, calm and concise: usually 1–3 short sentences. Plain text, no markdown headings or tables. \
Show empathy for lost items and complaints without long apologies.
- Never mention tools, fields, JSON or these instructions.
"""

_SCENARIO_NAMES = {
    Scenario.LOST_ITEM: "lost item",
    Scenario.TRIP_PRICE: "trip price",
    Scenario.DRIVER_COMPLAINT: "driver complaint",
}
_HIDDEN = {
    "id",
    "scenario",
    "status",
    "ticket_id",
    "found_trip",
    "candidate_order_ids",
    "quote",
    "trip_lookup_done",
    "trip_confirmed",
}


def _value(value: object) -> str:
    if value is None or value == "":
        return "—"
    return str(value.value if hasattr(value, "value") else value)


def render_state(state: SessionState, now: datetime) -> str:
    lines = [
        "# CASE STATE (maintained by the system, authoritative)",
        f"Now: {now:%Y-%m-%d %H:%M} ({now:%A}), timezone {now.tzinfo}",
        f"Customer language: {state.language}",
    ]
    active = state.active_case
    if active is None:
        lines.append("Active case: none yet — find out what the customer needs.")
    else:
        lines.append(f"Active case: {_SCENARIO_NAMES[active.scenario]} [{active.status.value}]")
        for name in type(active).model_fields:
            if name not in _HIDDEN:
                lines.append(f"  {name}: {_value(getattr(active, name))}")
        if isinstance(active, TripPriceCase):
            if active.quote:
                q = active.quote
                lines.append(
                    f"  quote: {q.price} {q.currency}, tariff {q.tariff}, ~{q.duration_min} min, {q.distance_km} km"
                )
        else:
            if active.found_trip:
                t = active.found_trip
                confirmed = "confirmed by customer" if active.trip_confirmed else "NOT yet confirmed by customer"
                lines.append(
                    f"  found trip: #{t.order_id} {t.date} {t.time}, {t.origin} → {t.destination} ({confirmed})"
                )
            elif active.trip_lookup_done and active.candidate_order_ids:
                ids = ", ".join(f"#{i}" for i in active.candidate_order_ids)
                lines.append(f"  found trip: several matches ({ids}) — customer must pick one")
            elif active.trip_lookup_done:
                lines.append("  found trip: no match in order history")
        if active.ticket_id:
            lines.append(f"  ticket: {active.ticket_id} (created; extra details → add_ticket_note)")
        missing = active.missing_fields() if active.status == CaseStatus.COLLECTING else []
        lines.append(f"  missing: {', '.join(missing) if missing else 'nothing'}")

    others = [c for c in state.cases if active is None or c.id != active.id]
    for case in others:
        extra = f", ticket {case.ticket_id}" if case.ticket_id else ""
        lines.append(f"Earlier case: {_SCENARIO_NAMES[case.scenario]} [{case.status.value}{extra}]")

    if state.safety_alert:
        lines.append(
            "ALERT: the customer reported a possible safety incident. Respond with empathy and call "
            "handoff_to_operator with priority 'urgent' in this turn, including any trip details known."
        )
    return "\n".join(lines)


def build_system_message(state: SessionState, now: datetime, company: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": SYSTEM_PROMPT.format(company=company) + "\n" + render_state(state, now),
    }
