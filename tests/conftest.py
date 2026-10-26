import pytest

from taxi_agent.config import Settings
from taxi_agent.factory import build_agent

from .helpers import NOW, FakeLLM


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, llm_api_key="test", _env_file=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def agent(settings, fake_llm):
    return build_agent(settings, llm=fake_llm, clock=lambda: NOW)
