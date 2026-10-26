"""QA suite: probes for weak spots. A red test means the weak spot is confirmed (see qa/PLAN.md).

Kept outside tests/ on purpose: `uv run pytest` stays green, `uv run pytest qa` runs the probes.
"""

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
from tests.helpers import KYIV, NOW, FakeLLM


def pytest_configure(config):
    config.addinivalue_line("markers", "crit: weak spot that can hurt a customer or lose an incident")
    config.addinivalue_line("markers", "major: core scenario or guardrail works wrong")
    config.addinivalue_line("markers", "minor: edge case, cost or robustness issue")


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
    kb = KnowledgeBase(Settings(_env_file=None).knowledge_base_dir)
    tickets = SQLiteTicketSink(SQLiteStore(tmp_path / "t.db"))

    def make(*user_messages: str, pricing=None, orders=None) -> ToolExecutor:
        ctx = ToolContext(
            state=SessionState(session_id="s1", created_at=NOW),
            messages=[{"role": "user", "content": m} for m in user_messages],
            now=NOW,
            orders=orders or StubOrderRepository(KYIV, "UAH", clock=lambda: NOW),
            pricing=pricing or StubPriceProvider("UAH"),
            tickets=tickets,
            kb=kb,
        )
        return ToolExecutor(ctx)

    return make
