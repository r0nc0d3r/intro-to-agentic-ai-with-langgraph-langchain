from learning_accelerator.observability.langfuse_setup import (
    get_langfuse_config,
    get_langfuse_handler,
    langfuse_enabled,
)


def test_langfuse_enabled_false_when_keys_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert langfuse_enabled() is False


def test_langfuse_enabled_false_when_only_public_key_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert langfuse_enabled() is False


def test_langfuse_enabled_true_when_both_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    assert langfuse_enabled() is True


def test_get_langfuse_handler_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert get_langfuse_handler() is None


def test_get_langfuse_handler_returns_handler_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:59999")

    handler = get_langfuse_handler()
    assert handler is not None


def test_get_langfuse_config_omits_callbacks_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    config = get_langfuse_config("session-123")

    assert config == {"configurable": {"thread_id": "session-123"}}
    assert "callbacks" not in config


def test_get_langfuse_config_includes_callbacks_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:59999")

    config = get_langfuse_config("session-456")

    assert config["configurable"]["thread_id"] == "session-456"
    assert len(config["callbacks"]) == 1
    assert config["metadata"] == {"langfuse_session_id": "session-456"}
