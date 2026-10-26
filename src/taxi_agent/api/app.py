from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from taxi_agent import __version__
from taxi_agent.agent.orchestrator import Agent, SessionNotFound
from taxi_agent.agent.state import CaseStatus, SessionState
from taxi_agent.api.schemas import (
    CostOut,
    EventOut,
    HistoryMessage,
    MessageIn,
    SessionHistoryOut,
    SessionOut,
    TurnOut,
)
from taxi_agent.config import PACKAGE_DIR, Settings, get_settings
from taxi_agent.factory import build_agent

WEB_DIR = PACKAGE_DIR / "web"


def create_app(settings: Settings | None = None, agent: Agent | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = agent or build_agent(settings)
        yield

    app = FastAPI(
        title="Taxi support agent",
        version=__version__,
        description="AI agent for lost items, trip price quotes and driver complaints.",
        lifespan=lifespan,
    )

    def get_agent(request: Request) -> Agent:
        return request.app.state.agent

    AgentDep = Annotated[Agent, Depends(get_agent)]

    async def session_payload(agent: Agent, state: SessionState) -> dict:
        case = state.active_case
        missing = case.missing_fields() if case and case.status == CaseStatus.COLLECTING else []
        usd = (
            state.prompt_tokens * settings.llm_price_input_per_m
            + state.completion_tokens * settings.llm_price_output_per_m
        ) / 1_000_000
        return {
            "session_id": state.session_id,
            "state": state,
            "missing_fields": missing,
            "tickets": await agent.tickets.list(state.session_id),
            "cost": CostOut(
                prompt_tokens=state.prompt_tokens,
                completion_tokens=state.completion_tokens,
                cached_prompt_tokens=state.cached_prompt_tokens,
                usd=round(usd, 6),
            ),
        }

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/api/sessions", status_code=status.HTTP_201_CREATED, response_model=SessionOut)
    async def create_session(agent: AgentDep):
        state = await agent.start_session()
        return await session_payload(agent, state)

    @app.get("/api/sessions/{session_id}", response_model=SessionHistoryOut)
    async def get_session(session_id: str, agent: AgentDep):
        loaded = await agent.store.load(session_id)
        if loaded is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        state, messages = loaded
        visible = [
            HistoryMessage(role=m["role"], content=m["content"])
            for m in messages
            if m["role"] in {"user", "assistant"} and m.get("content")
        ]
        return {**await session_payload(agent, state), "messages": visible}

    @app.post("/api/sessions/{session_id}/messages", response_model=TurnOut)
    async def post_message(session_id: str, body: MessageIn, agent: AgentDep):
        try:
            turn = await agent.handle_message(session_id, body.text)
        except SessionNotFound:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found") from None
        return {
            **await session_payload(agent, turn.state),
            "reply": turn.reply,
            "events": [EventOut(**asdict(e)) for e in turn.events],
        }

    @app.get("/api/sessions/{session_id}/tickets")
    async def list_tickets(session_id: str, agent: AgentDep):
        return await agent.tickets.list(session_id)

    @app.get("/approach", include_in_schema=False)
    async def approach():
        if not settings.approach_page.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "approach page is not built")
        return FileResponse(settings.approach_page, media_type="text/html")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
