import uuid

from learning_accelerator.mcp_servers.memory_server import (
    memory_delete,
    memory_get,
    memory_list_keys,
    memory_set,
)


def _session_id() -> str:
    return str(uuid.uuid4())


def test_memory_set_and_get_roundtrip():
    session_id = _session_id()
    memory_set(session_id, "goal", "learn langgraph")
    assert memory_get(session_id, "goal") == "learn langgraph"


def test_memory_get_missing_key_returns_null_string():
    session_id = _session_id()
    assert memory_get(session_id, "missing") == "null"


def test_memory_list_keys():
    session_id = _session_id()
    memory_set(session_id, "a", "1")
    memory_set(session_id, "b", "2")
    assert memory_list_keys(session_id) == ["a", "b"]


def test_memory_delete():
    session_id = _session_id()
    memory_set(session_id, "a", "1")
    memory_delete(session_id, "a")
    assert memory_get(session_id, "a") == "null"


def test_memory_isolated_between_sessions():
    session_a = _session_id()
    session_b = _session_id()
    memory_set(session_a, "key", "value-a")
    assert memory_get(session_b, "key") == "null"
