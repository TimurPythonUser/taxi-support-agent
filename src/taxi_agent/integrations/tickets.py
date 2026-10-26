from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel


class TicketDraft(BaseModel):
    """What the agent hands to a human operator."""

    session_id: str
    kind: str  # lost_item | driver_complaint | handoff
    priority: str  # normal | urgent
    summary: str  # short Russian summary for the operator
    fields: dict[str, Any]  # structured data collected from the customer (built by code, not by the LLM)
    customer_language: str
    transcript: list[dict[str, str]]  # last visible messages, for context


class Ticket(TicketDraft):
    id: str
    created_at: datetime
    notes: list[str] = []  # details the customer added after the ticket was created


class TicketSink(Protocol):
    async def create(self, draft: TicketDraft) -> Ticket: ...

    async def add_note(self, ticket_id: str, note: str) -> None: ...

    async def list(self, session_id: str | None = None) -> list[Ticket]: ...
