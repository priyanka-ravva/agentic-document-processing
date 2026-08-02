"""Tests for runtime LLM configuration."""

import importlib
import sys
import types


class FakeChatGroq:
    """Capture ChatGroq constructor kwargs for tests."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_get_llm_sets_configured_temperature(monkeypatch) -> None:
    """Groq clients should use a low extraction-friendly temperature by default."""

    langchain_groq = types.ModuleType("langchain_groq")
    langchain_groq.ChatGroq = FakeChatGroq
    monkeypatch.setitem(sys.modules, "langchain_groq", langchain_groq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_TEMPERATURE", raising=False)

    import src.config as config

    config = importlib.reload(config)
    config.get_settings.cache_clear()

    llm = config.get_llm()

    assert llm.kwargs["temperature"] == 0.1
