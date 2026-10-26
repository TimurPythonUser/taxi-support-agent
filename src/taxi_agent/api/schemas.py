from typing import Any

from pydantic import BaseModel, Field

from taxi_agent.agent.state import SessionState
from taxi_agent.integrations.tickets import Ticket


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class EventOut(BaseModel):
    kind: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ms: int


class CostOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    usd: float


class SessionOut(BaseModel):
    session_id: str
    state: SessionState
    missing_fields: list[str]
    tickets: list[Ticket]
    cost: CostOut


class TurnOut(SessionOut):
    reply: str
    events: list[EventOut]


class HistoryMessage(BaseModel):
    role: str
    content: str


class SessionHistoryOut(SessionOut):
    messages: list[HistoryMessage]
