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

    model_config = {  # pydantic v2 style
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
