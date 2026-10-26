import pytest

from taxi_agent.agent.state import SessionState
from taxi_agent.agent.tools import ToolContext, ToolExecutor
from taxi_agent.config import Settings
from taxi_agent.factory import build_agent
from taxi_agent.integrations.orders import StubOrderRepository
from taxi_agent.integrations.pricing import StubPriceProvider
from taxi_agent.kb import KnowledgeBase
from taxi_agent.storage import SQLiteStore
from taxi_agent.storage.sqlite import SQLiteTicketSink

from .helpers import KYIV, NOW, FakeLLM


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, llm_api_key="test", _env_file=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def agent(settings, fake_llm):
    return build_agent(settings, llm=fake_llm, clock=lambda: NOW)


@pytest.fixture
def make_executor(tmp_path):
    """A tool executor over a fresh session whose customer has written `user_messages`."""
    kb = KnowledgeBase(Settings(_env_file=None).knowledge_base_dir)
    tickets = SQLiteTicketSink(SQLiteStore(tmp_path / "t.db"))

    def make(*user_messages: str, pricing=None) -> ToolExecutor:
        ctx = ToolContext(
            state=SessionState(session_id="s1", created_at=NOW),
            messages=[{"role": "user", "content": m} for m in user_messages],
            now=NOW,
            orders=StubOrderRepository(KYIV, "UAH", clock=lambda: NOW),
            pricing=pricing or StubPriceProvider("UAH"),
            tickets=tickets,
            kb=kb,
        )
        return ToolExecutor(ctx)

    return make
