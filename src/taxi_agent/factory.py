"""Wiring: the only place that knows which concrete adapters are used."""

from taxi_agent.agent.orchestrator import Agent
from taxi_agent.config import Settings
from taxi_agent.integrations.orders import OrderRepository, StubOrderRepository
from taxi_agent.integrations.pricing import PriceProvider, StubPriceProvider
from taxi_agent.kb import KnowledgeBase
from taxi_agent.llm import LLMClient, OpenAICompatClient
from taxi_agent.storage import SQLiteStore
from taxi_agent.storage.sqlite import SQLiteTicketSink


def build_agent(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    pricing: PriceProvider | None = None,
    orders: OrderRepository | None = None,
    clock=None,
) -> Agent:
    store = SQLiteStore(settings.db_path)
    return Agent(
        settings=settings,
        llm=llm or OpenAICompatClient(settings),
        store=store,
        tickets=SQLiteTicketSink(store),
        orders=orders or StubOrderRepository(settings.tz, settings.currency, clock=clock),
        pricing=pricing or StubPriceProvider(settings.currency),
        kb=KnowledgeBase(settings.knowledge_base_dir),
        clock=clock,
    )
