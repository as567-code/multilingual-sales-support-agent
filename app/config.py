"""Runtime settings — loaded from environment (.env) via pydantic-settings.

Single source of truth for model names, index paths, and retrieval knobs.
Imported by every agent so tests and runtime share one configuration.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    llm_provider: str = Field(default="mistral", alias="LLM_PROVIDER")
    llm_model: str = Field(default="mistral-small-latest", alias="LLM_MODEL")

    # --- Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- Retrieval
    embed_model: str = "intfloat/multilingual-e5-base"
    index_dir: Path = Path("data/processed")
    # Unified multilingual index (EN+ES+HI rows in one store)
    index_file: str = "faiss_all.index"
    meta_file: str = "metadata.jsonl"

    retrieval_top_k: int = 5
    # Cosine-similarity floor for e5 on this corpus. Smoke-tested hits sit
    # 0.81–0.88; 0.70 cleanly separates real matches from OOD noise.
    retrieval_min_score: float = 0.70


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
