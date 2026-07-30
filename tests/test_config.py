import pytest

from learning_accelerator.config import get_chat_model


def test_get_chat_model_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    model = get_chat_model()
    assert type(model).__name__ == "ChatOllama"


def test_get_chat_model_ollama_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:12b-mlx")
    model = get_chat_model()
    assert model.model == "gemma4:12b-mlx"


def test_get_chat_model_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = get_chat_model()
    assert type(model).__name__ == "ChatAnthropic"


def test_get_chat_model_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = get_chat_model()
    assert type(model).__name__ == "ChatOpenAI"


def test_get_chat_model_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_chat_model()


def test_get_chat_model_caches_same_temperature(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    model1 = get_chat_model(temperature=0.2)
    model2 = get_chat_model(temperature=0.2)
    assert model1 is model2


def test_get_chat_model_different_temperature_not_cached(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    model1 = get_chat_model(temperature=0.2)
    model2 = get_chat_model(temperature=0.3)
    assert model1 is not model2
