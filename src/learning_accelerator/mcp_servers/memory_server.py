from __future__ import annotations

from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Memory Server")

_store: dict[str, dict[str, dict[str, str]]] = {}


@mcp.tool()
def memory_set(session_id: str, key: str, value: str) -> str:
    """Store a string value under key, scoped to session_id."""
    session = _store.setdefault(session_id, {})
    session[key] = {
        "value": value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return "ok"


@mcp.tool()
def memory_get(session_id: str, key: str) -> str:
    """Retrieve a stored value by key for session_id.

    Returns the string "null" (not Python None) if the key is missing —
    avoids None-handling edge cases in LLM tool output.
    """
    entry = _store.get(session_id, {}).get(key)
    return entry["value"] if entry else "null"


@mcp.tool()
def memory_list_keys(session_id: str) -> list[str]:
    """List all keys stored for session_id."""
    return sorted(_store.get(session_id, {}).keys())


@mcp.tool()
def memory_delete(session_id: str, key: str) -> str:
    """Delete a key for session_id. Returns 'ok' whether or not it existed."""
    _store.get(session_id, {}).pop(key, None)
    return "ok"


@mcp.resource("notes://session/{session_id}")
def session_summary(session_id: str) -> str:
    """Markdown summary of everything stored for a session."""
    session = _store.get(session_id, {})
    if not session:
        return f"# Session {session_id}\n\n(no stored data)"
    lines = [f"# Session {session_id}", ""]
    for key, entry in sorted(session.items()):
        lines.append(f"- **{key}**: {entry['value']} (updated {entry['updated_at']})")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
