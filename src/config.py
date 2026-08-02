"""Application configuration and model client setup."""

from functools import lru_cache
from typing import Optional

from langchain_groq import ChatGroq
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_fallback_models: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_FALLBACK_MODELS",
    )
    groq_temperature: float = Field(default=0.1, alias="GROQ_TEMPERATURE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


def get_model_names() -> list[str]:
    """Return primary and fallback Groq model names in priority order."""

    settings = get_settings()
    fallback_models = [
        model.strip()
        for model in settings.groq_fallback_models.split(",")
        if model.strip()
    ]

    model_names = [settings.groq_model, *fallback_models]
    return list(dict.fromkeys(model_names))


def get_llm(model_name: Optional[str] = None) -> ChatGroq:
    """Create the Groq chat model client.

    The client is created only when called, so importing the project does not
    require a configured API key.
    """

    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Copy .env.example to .env and add your Groq API key."
        )

    return ChatGroq(
        model_name=model_name or settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=settings.groq_temperature,
    )
