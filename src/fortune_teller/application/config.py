"""Application settings — all config via environment variables or .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for Fortune Teller.

    All fields can be overridden by environment variables (no prefix needed)
    or by a .env file in the project root.
    """

    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_api_key: str = "sk-no-key"
    chat_model: str = "local-model"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model_path: Path = Path("./data/models/bge-small-en-v1.5")
    ft_data_dir: Path = Path("./data")
    sqlite_path: Path = Path("./data/sqlite/fortune.db")
    images_dir: Path = Path("./data/images")
    normalize_provider: str = "api"  # "api" for Claude, "local" for llama-server
    normalize_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # set via env var ANTHROPIC_API_KEY
    per_card_timeout: float = 60.0
    summary_timeout_base: float = 120.0
    summary_timeout_per_card: float = 12.0

    model_config = {  # pydantic v2 style
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


def summary_timeout(n_positions: int) -> float:
    """Compute the summary-chain HTTP timeout for a spread of *n_positions*.

    The summary prompt grows linearly with the number of card interpretations,
    so the timeout must scale accordingly to avoid spurious failures on large
    spreads (e.g. 10-card Celtic Cross) when the local LLM runs on CPU.

    Args:
        n_positions: Number of positions in the spread.

    Returns:
        ``summary_timeout_base + summary_timeout_per_card * n_positions``.
    """
    return settings.summary_timeout_base + settings.summary_timeout_per_card * n_positions


settings = Settings()
