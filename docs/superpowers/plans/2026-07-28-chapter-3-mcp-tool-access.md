# Chapter 3: Standardized Tool Access with MCP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the filesystem and memory MCP servers (FastMCP), sample study
materials for them to serve, and the Explainer agent that calls their tools
in an iterative loop — then wire it into the graph after Curriculum Planner.

**Architecture:** `mcp_servers/filesystem_server.py` and
`mcp_servers/memory_server.py` expose plain Python functions decorated with
`@mcp.tool()`/`@mcp.resource()` (runnable standalone via `mcp.run()`, stdio
transport by default). `agents/explainer.py` imports those functions
directly and re-wraps each with LangChain's `@tool` decorator — the
article's own documented dev-mode shortcut (in-process, no subprocess/client
transport yet; that's a noted production upgrade via
`MultiServerMCPClient`, out of scope here). `graph/workflow.py` gains a
second node: `curriculum_planner → explainer → END`.

**Tech Stack:** `mcp` (FastMCP), LangChain `@tool`/`.bind_tools()`, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-3` (already created, off updated `main`)
- One PR per chapter, merged before chapter 4 starts
- Confirmed via context7: `from mcp.server.fastmcp import FastMCP` is the
  current stable import (a `MCPServer` rename exists only in a v2 alpha,
  not used here)
- `memory_get` must return the string `"null"` for a missing key, not
  Python `None` — deliberate, avoids `None`-handling edge cases in LLM
  tool output (per article)
- `read_study_file` must reject path traversal — resolve both the notes
  base directory and the requested path, reject if the requested path
  isn't inside the base directory
- Testing approach (from spec): pytest for pure logic. The MCP server tool
  functions qualify (file I/O / dict ops, no LLM calls) — full coverage
  expected here. `explainer_node` itself (LLM tool-calling loop) stays
  manual-run only, verified against local Ollama
- Local dev machine has Ollama with `gemma4:12b-mlx`, `gemma4:12b`,
  `qwen3.5:2b` — chapter 2 found `gemma4:12b` works for structured output
  where `gemma4:12b-mlx` doesn't; verify tool-calling support per-model
  here too rather than assuming the same result carries over
- `learning/chapterN.md` format: short Q/A flashcards (see
  `learning/chapter1.md` and `learning/chapter2.md`)

---

### Task 1: Add the `mcp` dependency and sample study materials

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Create: `study_materials/sample_notes/langgraph_basics.md`
- Create: `study_materials/sample_notes/state_management.md`

**Interfaces:**
- Produces: two markdown files under `study_materials/sample_notes/` that
  Task 2's filesystem server (default `NOTES_PATH`) and Task 6's demo
  script will read

- [ ] **Step 1: Add the mcp package**

```bash
uv add mcp
```

Expected: `pyproject.toml` gains an `mcp` dependency; `uv.lock` updates.

- [ ] **Step 2: Write `study_materials/sample_notes/langgraph_basics.md`**

```markdown
# LangGraph Basics

LangGraph models an agent as a graph of nodes and edges over a shared
state object. Each node is a plain function that reads the current state
and returns a partial update — LangGraph merges that update back in.

Key ideas:

- **Nodes** are functions: `(state) -> dict`.
- **Edges** connect nodes and can be conditional (a routing function
  decides which node runs next based on the current state).
- **State** is typically a `TypedDict` describing every field any node
  might read or write.
- **Checkpointing** persists state after each step, so a run can be
  paused, resumed, or recovered after a crash.

A graph always has a `START` and at least one path to `END`.
```

- [ ] **Step 3: Write `study_materials/sample_notes/state_management.md`**

```markdown
# State Management in LangGraph

State fields update in one of two ways:

1. **Last-write-wins** — a node's returned value for a key simply
   replaces the previous value. Most fields work this way (e.g. a
   `roadmap` or an `approved` flag).
2. **Reducer-based** — a field is annotated with a reducer function that
   controls how new values combine with old ones. The most common
   example is `messages`, annotated with `add_messages`, which appends
   new messages instead of replacing the list.

Nodes should return only the keys they actually changed — not the whole
state — so unrelated fields are left untouched by that step.
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock study_materials/
git commit -m "Add mcp dependency and sample study materials"
```

---

### Task 2: Filesystem MCP server + tests

**Files:**
- Create: `src/learning_accelerator/mcp_servers/__init__.py`
- Create: `src/learning_accelerator/mcp_servers/filesystem_server.py`
- Test: `tests/test_filesystem_server.py`

**Interfaces:**
- Consumes: nothing
- Produces: `list_study_files() -> list[str]`,
  `read_study_file(filename: str) -> str`,
  `search_notes(query: str) -> list[str]` — imported directly by
  `agents/explainer.py` in Task 4

- [ ] **Step 1: Write the failing tests**

Create `tests/test_filesystem_server.py`:

```python
import pytest

from learning_accelerator.mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
)


def test_list_study_files_sorted(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "b.md").write_text("second")
    (tmp_path / "a.md").write_text("first")

    assert list_study_files() == ["a.md", "b.md"]


def test_list_study_files_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path / "does-not-exist"))
    assert list_study_files() == []


def test_read_study_file_returns_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "note.md").write_text("hello world")

    assert read_study_file("note.md") == "hello world"


def test_read_study_file_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    assert read_study_file("missing.md") == "File not found: missing.md"


def test_read_study_file_blocks_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path.parent / "secret.md").write_text("top secret")

    with pytest.raises(ValueError, match="outside the notes directory"):
        read_study_file("../secret.md")


def test_search_notes_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "note.md").write_text("LangGraph is great\nOther line")

    assert search_notes("langgraph") == ["note.md: LangGraph is great"]


def test_search_notes_caps_at_20_results(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "note.md").write_text("\n".join(f"match {i}" for i in range(30)))

    assert len(search_notes("match")) == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_filesystem_server.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.mcp_servers'`

- [ ] **Step 3: Create `src/learning_accelerator/mcp_servers/__init__.py`**

```python
```

(empty file)

- [ ] **Step 4: Write `src/learning_accelerator/mcp_servers/filesystem_server.py`**

```python
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Filesystem Server")


def _notes_base() -> Path:
    return Path(os.getenv("NOTES_PATH", "study_materials/sample_notes")).resolve()


def _resolve_safe(filename: str) -> Path:
    base = _notes_base()
    candidate = (base / filename).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"'{filename}' is outside the notes directory")
    return candidate


@mcp.tool()
def list_study_files() -> list[str]:
    """List available study note filenames, sorted alphabetically."""
    base = _notes_base()
    if not base.exists():
        return []
    return sorted(p.name for p in base.glob("*.md"))


@mcp.tool()
def read_study_file(filename: str) -> str:
    """Read the contents of a study note by filename."""
    path = _resolve_safe(filename)
    if not path.exists():
        return f"File not found: {filename}"
    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_notes(query: str) -> list[str]:
    """Case-insensitive substring search across study notes.

    Returns up to 20 matches formatted as 'filename: line'.
    """
    base = _notes_base()
    results: list[str] = []
    if not base.exists():
        return results
    needle = query.lower()
    for path in sorted(base.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if needle in line.lower():
                results.append(f"{path.name}: {line.strip()}")
                if len(results) >= 20:
                    return results
    return results


@mcp.resource("notes://index")
def notes_index() -> str:
    """Markdown index of available study materials with file sizes."""
    base = _notes_base()
    if not base.exists():
        return "# Study Materials\n\n(no notes directory found)"
    lines = ["# Study Materials", ""]
    for path in sorted(base.glob("*.md")):
        lines.append(f"- {path.name} ({path.stat().st_size} bytes)")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_filesystem_server.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/learning_accelerator/mcp_servers/__init__.py src/learning_accelerator/mcp_servers/filesystem_server.py tests/test_filesystem_server.py
git commit -m "Add filesystem MCP server"
```

---

### Task 3: Memory MCP server + tests

**Files:**
- Create: `src/learning_accelerator/mcp_servers/memory_server.py`
- Test: `tests/test_memory_server.py`

**Interfaces:**
- Consumes: nothing
- Produces: `memory_set(session_id, key, value) -> str`,
  `memory_get(session_id, key) -> str`,
  `memory_list_keys(session_id) -> list[str]`,
  `memory_delete(session_id, key) -> str` — imported directly by
  `agents/explainer.py` in Task 4

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory_server.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.mcp_servers.memory_server'`

- [ ] **Step 3: Write `src/learning_accelerator/mcp_servers/memory_server.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory_server.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/mcp_servers/memory_server.py tests/test_memory_server.py
git commit -m "Add memory MCP server"
```

---

### Task 4: Explainer agent

**Files:**
- Create: `src/learning_accelerator/agents/explainer.py`

**Interfaces:**
- Consumes: `list_study_files`/`read_study_file`/`search_notes` (Task 2),
  `memory_get`/`memory_set` (Task 3), `get_chat_model` (chapter 2),
  `AgentState` (chapter 2)
- Produces: `explainer_node(state: AgentState) -> dict`, wired into the
  graph in Task 5

No unit test here — this node makes live LLM tool-calling loop, excluded
from pytest per the spec's testing approach. Verified for real in Task 6.

- [ ] **Step 1: Write `src/learning_accelerator/agents/explainer.py`**

```python
from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import AgentState
from learning_accelerator.mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
)
from learning_accelerator.mcp_servers.memory_server import memory_get, memory_set

MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are the Explainer. Use the available tools to read \
study materials and explain the current topic clearly to the learner. \
Save a short summary to memory under the key "last_explanation" when \
you're done. Stop calling tools once you've given your final explanation."""


@tool
def tool_list_files() -> list[str]:
    """List available study note filenames."""
    return list_study_files()


@tool
def tool_read_file(filename: str) -> str:
    """Read a study note by filename."""
    return read_study_file(filename)


@tool
def tool_search_notes(query: str) -> list[str]:
    """Search study notes for a substring, case-insensitive."""
    return search_notes(query)


@tool
def tool_memory_get(session_id: str, key: str) -> str:
    """Retrieve a value from session memory."""
    return memory_get(session_id, key)


@tool
def tool_memory_set(session_id: str, key: str, value: str) -> str:
    """Store a value in session memory."""
    return memory_set(session_id, key, value)


EXPLAINER_TOOLS = [
    tool_list_files,
    tool_read_file,
    tool_search_notes,
    tool_memory_get,
    tool_memory_set,
]

_TOOLS_BY_NAME = {t.name: t for t in EXPLAINER_TOOLS}


def _execute_tool_call(tool_call: dict) -> str:
    tool_fn = _TOOLS_BY_NAME[tool_call["name"]]
    result = tool_fn.invoke(tool_call["args"])
    return str(result)


def explainer_node(state: AgentState) -> dict:
    llm = get_chat_model(temperature=0.3).bind_tools(EXPLAINER_TOOLS)

    topic = state["roadmap"].topics[state["current_topic_index"]]
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                f"Session ID: {state['session_id']}. "
                f"Current topic: {topic.title} — {topic.description}"
            )
        ),
    ]

    final_response = None
    for _ in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            final_response = response
            break

        for tool_call in response.tool_calls:
            result = _execute_tool_call(tool_call)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    if final_response is None:
        return {"error": "explainer exceeded max iterations"}

    return {"messages": [final_response], "error": None}
```

- [ ] **Step 2: Commit**

```bash
git add src/learning_accelerator/agents/explainer.py
git commit -m "Add Explainer agent with iterative MCP tool-calling loop"
```

---

### Task 5: Wire Explainer into the graph

**Files:**
- Modify: `src/learning_accelerator/graph/workflow.py`

**Interfaces:**
- Consumes: `explainer_node` (Task 4)
- Produces: graph now runs `START → curriculum_planner → explainer → END`

- [ ] **Step 1: Edit `src/learning_accelerator/graph/workflow.py`**

Add the import:

```python
from learning_accelerator.agents.explainer import explainer_node
```

Replace:

```python
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", END)
```

with:

```python
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("explainer", explainer_node)
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "explainer")
    builder.add_edge("explainer", END)
```

- [ ] **Step 2: Verify the graph still compiles**

Run: `rm -f .data/checkpoints.sqlite && uv run python -c "from learning_accelerator.graph.workflow import graph; print(graph)"`
Expected: prints a `CompiledStateGraph` object, no errors

- [ ] **Step 3: Run the full pytest suite**

Run: `uv run pytest -v`
Expected: all existing tests (state, config, filesystem_server,
memory_server) still pass — this task only changes graph wiring, no new
pytest tests.

- [ ] **Step 4: Commit**

```bash
git add src/learning_accelerator/graph/workflow.py
git commit -m "Wire Explainer into graph after Curriculum Planner"
```

---

### Task 6: Manual demo script + real verification run

**Files:**
- Create: `scripts/demo_chapter3.py`

**Interfaces:**
- Consumes: `graph`, `initial_state` from chapter 2/Task 5

- [ ] **Step 1: Write `scripts/demo_chapter3.py`**

```python
"""Manual run: invoke the chapter 3 graph (Curriculum Planner + Explainer).

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly. The
Explainer additionally needs a model that supports tool-calling.
"""

from __future__ import annotations

import uuid

from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import graph


def main() -> None:
    session_id = str(uuid.uuid4())
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)

    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(state, config=config)

    print(f"Session: {session_id}")
    print(f"Roadmap topics: {[t.title for t in result['roadmap'].topics]}")
    print(f"Explainer output:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real against local Ollama**

```bash
rm -f .data/checkpoints.sqlite
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b uv run python scripts/demo_chapter3.py
```

Expected: prints a session ID, the roadmap's topic titles, and a final
explanation. If the Explainer's tool-calling loop errors or never calls
`tool_read_file`/`tool_list_files` meaningfully, try `qwen3.5:2b` as an
alternative and note in the chapter 3 flashcards which model(s) actually
exercised the tools correctly.

- [ ] **Step 3: Run the full pytest suite once more**

Run: `uv run pytest -v`
Expected: all tests still pass (this step only adds a script).

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_chapter3.py
git commit -m "Add chapter 3 manual demo script"
```

---

### Task 7: Chapter 3 learning flashcards

**Files:**
- Create: `learning/chapter3.md`

- [ ] **Step 1: Write `learning/chapter3.md`**

```markdown
## Chapter 3: Standardized Tool Access with MCP

**Q: What are the three primitives MCP defines?**
A: Tools (executable actions), Resources (read-only data by URI), and
Prompts (reusable prompt templates owned by the server).

**Q: Why does `memory_get` return the string `"null"` instead of Python
`None` for a missing key?**
A: To avoid `None`-handling edge cases when the result flows into LLM
tool output — a plain string is unambiguous everywhere it gets used.

**Q: How does `read_study_file` prevent path traversal?**
A: It resolves both the notes base directory and the requested file path
to absolute paths, then rejects the request unless the resolved file
path is actually inside the resolved base directory.

**Q: How does the Explainer agent talk to the MCP servers in this
chapter — via a real client/server connection, or something simpler?**
A: Something simpler (and this is deliberate, per the article): it
imports the server's plain Python functions directly and wraps each with
LangChain's `@tool` decorator, all in one process. A production setup
would swap in `MultiServerMCPClient` over a real subprocess transport —
the agent-facing tool-calling code doesn't change either way.

**Q: What temperature does the Explainer use, and why?**
A: `0.3` — balances multi-turn tool-calling reasoning with enough
consistency to stay on-task, unlike the Curriculum Planner's `0.1`.

**Q: What ends the Explainer's tool-calling loop?**
A: The LLM's response has no `tool_calls` (it gave a final explanation
instead of requesting another tool), or 8 iterations are reached,
whichever comes first.

**Q: Why must `ToolMessage.tool_call_id` match the id from the LLM's
`tool_calls` request?**
A: The LLM correlates each tool result back to the specific call it made
by that id — without a match, it can't tell which result answers which
request.

**Q: What does chapter 3's graph look like now?**
A: `START → curriculum_planner → explainer → END`. `human_approval`
still doesn't exist — it's inserted between them in chapter 5.

**Q: Which local Ollama model(s) worked for the Explainer's tool-calling
loop?**
A: See the demo script run output for this repo — record whichever
model(s) actually invoked `tool_list_files`/`tool_read_file` correctly
versus which ones failed to produce valid tool calls.
```

Before committing, fill in the last flashcard's answer with what Task 6
actually observed.

- [ ] **Step 2: Commit**

```bash
git add learning/chapter3.md
git commit -m "Add chapter 3 learning flashcards"
```

---

### Task 8: Open the chapter 3 PR

**Files:** none (branch/PR operation only)

- [ ] **Step 1: Confirm all tests pass**

Run: `uv run pytest -v`
Expected: all tests pass (state, config, filesystem_server, memory_server).

- [ ] **Step 2: Push the branch**

**Before running this step, ask the user to confirm.**

```bash
git push -u origin agent/chapter-3
```

- [ ] **Step 3: Open the PR**

**Before running this step, ask the user to confirm.**

```bash
gh pr create --title "Chapter 3: Standardized Tool Access with MCP" --body "$(cat <<'EOF'
## Summary
- Add filesystem MCP server (list/read/search study notes, path-traversal
  guarded) and memory MCP server (session-scoped key/value store)
- Add sample study materials under study_materials/sample_notes/
- Add the Explainer agent: imports server functions directly, wraps with
  @tool, iterative tool-calling loop (temperature=0.3, max 8 iterations)
- Wire Explainer into the graph: curriculum_planner -> explainer -> END
- Add learning/chapter3.md flashcards

## Test plan
- [ ] `uv sync && uv run pytest -v` — all tests pass (state, config,
      filesystem_server, memory_server)
- [ ] `uv run python scripts/demo_chapter3.py` — prints a roadmap and a
      final Explainer explanation

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report the PR URL to the user**
