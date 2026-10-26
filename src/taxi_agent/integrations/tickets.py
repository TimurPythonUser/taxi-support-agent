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


class TicketSink(Protocol):
    async def create(self, draft: TicketDraft) -> Ticket: ...

    async def list(self, session_id: str | None = None) -> list[Ticket]: ...
