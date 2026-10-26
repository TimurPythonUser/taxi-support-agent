"""SQLite persistence for the prototype: sessions (state + message history), tickets, daily usage.

Behind small Protocols, so production can move sessions to Redis/Postgres and tickets
to the platform's helpdesk API without touching the agent.
"""

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from taxi_agent.agent.state import SessionState
from taxi_agent.integrations.tickets import Ticket, TicketDraft

Message = dict[str, Any]


class SessionStore(Protocol):
    async def create(self, state: SessionState) -> None: ...

    async def load(self, session_id: str) -> tuple[SessionState, list[Message]] | None: ...

    async def save(self, state: SessionState, messages: list[Message]) -> None: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    messages TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tickets_session ON tickets(session_id);
CREATE TABLE IF NOT EXISTS daily_usage (
    day TEXT PRIMARY KEY,
    requests INTEGER NOT NULL
);
"""


class SQLiteStore:
    """Implements SessionStore and TicketSink. One short-lived connection per call keeps it thread-safe."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # --- sessions ---------------------------------------------------------------------------
    async def create(self, state: SessionState) -> None:
        await self._run(self._create, state)

    def _create(self, state: SessionState) -> None:
        now = datetime.now().isoformat()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO sessions (id, state, messages, created_at, updated_at) VALUES (?, ?, '[]', ?, ?)",
                (state.session_id, state.model_dump_json(), now, now),
            )

    async def load(self, session_id: str) -> tuple[SessionState, list[Message]] | None:
        return await self._run(self._load, session_id)

    def _load(self, session_id: str) -> tuple[SessionState, list[Message]] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT state, messages FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return SessionState.model_validate_json(row["state"]), json.loads(row["messages"])

    async def save(self, state: SessionState, messages: list[Message]) -> None:
        await self._run(self._save, state, messages)

    def _save(self, state: SessionState, messages: list[Message]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE sessions SET state = ?, messages = ?, updated_at = ? WHERE id = ?",
                (
                    state.model_dump_json(),
                    json.dumps(messages, ensure_ascii=False),
                    datetime.now().isoformat(),
                    state.session_id,
                ),
            )

    # --- tickets ----------------------------------------------------------------------------
    async def create_ticket(self, draft: TicketDraft) -> Ticket:
        return await self._run(self._create_ticket, draft)

    def _create_ticket(self, draft: TicketDraft) -> Ticket:
        created_at = datetime.now().astimezone()
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO tickets (session_id, created_at, body) VALUES (?, ?, '{}')",
                (draft.session_id, created_at.isoformat()),
            )
            ticket = Ticket(id=f"OP-{cur.lastrowid:05d}", created_at=created_at, **draft.model_dump())
            conn.execute(
                "UPDATE tickets SET id = ?, body = ? WHERE seq = ?",
                (ticket.id, ticket.model_dump_json(), cur.lastrowid),
            )
        return ticket

    async def add_ticket_note(self, ticket_id: str, note: str) -> None:
        await self._run(self._add_ticket_note, ticket_id, note)

    def _add_ticket_note(self, ticket_id: str, note: str) -> None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT body FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None:
                raise KeyError(ticket_id)
            ticket = Ticket.model_validate_json(row["body"])
            ticket.notes.append(note)
            conn.execute("UPDATE tickets SET body = ? WHERE id = ?", (ticket.model_dump_json(), ticket_id))

    async def list_tickets(self, session_id: str | None = None) -> list[Ticket]:
        return await self._run(self._list_tickets, session_id)

    def _list_tickets(self, session_id: str | None) -> list[Ticket]:
        query, args = "SELECT body FROM tickets ORDER BY seq DESC LIMIT 200", ()
        if session_id:
            query, args = "SELECT body FROM tickets WHERE session_id = ? ORDER BY seq", (session_id,)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, args).fetchall()
        return [Ticket.model_validate_json(r["body"]) for r in rows]

    # --- demo cost guard --------------------------------------------------------------------
    async def count_request(self, day: date) -> int:
        """Increment and return today's number of LLM-backed requests."""
        return await self._run(self._count_request, day)

    def _count_request(self, day: date) -> int:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO daily_usage (day, requests) VALUES (?, 1) "
                "ON CONFLICT(day) DO UPDATE SET requests = requests + 1",
                (day.isoformat(),),
            )
            return conn.execute("SELECT requests FROM daily_usage WHERE day = ?", (day.isoformat(),)).fetchone()[0]


class SQLiteTicketSink:
    """TicketSink view over SQLiteStore."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    async def create(self, draft: TicketDraft) -> Ticket:
        return await self._store.create_ticket(draft)

    async def add_note(self, ticket_id: str, note: str) -> None:
        await self._store.add_ticket_note(ticket_id, note)

    async def list(self, session_id: str | None = None) -> list[Ticket]:
        return await self._store.list_tickets(session_id)
