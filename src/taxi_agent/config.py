from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """All runtime configuration comes from environment variables or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    # Any OpenAI-compatible Chat Completions endpoint. DeepSeek is the default.
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    llm_api_key: str = Field(default="", validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"))
    # Provider-specific request fields. DeepSeek: disable reasoning — the task needs reliable
    # tool calls and short answers, not long chains of thought.
    llm_extra_body: dict[str, Any] = {"thinking": {"type": "disabled"}}
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700
    llm_timeout_s: float = 40.0
    llm_max_retries: int = 2
    # USD per 1M tokens, only used to show the cost of a dialog in the UI.
    llm_price_input_per_m: float = 0.14
    llm_price_output_per_m: float = 0.28

    timezone: str = "Europe/Kyiv"
    currency: str = "UAH"
    company_name: str = "Opti Taxi"

    data_dir: Path = Path("data")
    knowledge_base_dir: Path = PACKAGE_DIR / "knowledge_base"
    approach_page: Path = Path("docs/approach.html")

    history_window_user_turns: int = 12
    max_tool_iterations: int = 6
    max_messages_per_session: int = 40
    daily_request_limit: int = 2000
    stall_turns_before_handoff: int = 3

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "taxi_agent.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
