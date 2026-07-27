# Chapter 2: Stateful Orchestration with LangGraph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the project with `uv`, define the shared `AgentState` schema,
implement the provider-agnostic LLM config switch, build the Curriculum
Planner node, and wire the first LangGraph graph with SQLite checkpointing.

**Architecture:** src-layout package `learning_accelerator` under `src/`.
`config.py` abstracts provider selection (Ollama/Anthropic/OpenAI) behind
`get_chat_model()`. `graph/state.py` defines `AgentState` (TypedDict) plus
`Topic`/`StudyRoadmap` (Pydantic, for cross-provider `with_structured_output`).
`agents/curriculum_planner.py` implements the one node this chapter adds.
`graph/workflow.py` wires `START → curriculum_planner → END` with a
persistent (non-context-manager) `SqliteSaver` checkpointer — the pattern
confirmed current against the `langgraph-checkpoint-sqlite` source via
context7.

**Tech Stack:** Python 3.11+, `uv`, LangGraph, LangChain (core + ollama +
anthropic + openai partner packages), Pydantic, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-2` (already created, off updated `main`)
- One PR per chapter, merged before chapter 3 starts
- `LLM_PROVIDER` env var selects `ollama` (default) / `anthropic` / `openai` — must work for all three without code changes, only env changes
- Local dev machine has Ollama running with models `gemma4:12b-mlx`, `gemma4:12b`, `qwen3.5:2b` available — use `gemma4:12b-mlx` as the `.env.example` default
- Testing approach (from spec): pytest only for pure logic (state defaults, provider selection); LLM-calling nodes verified by manual run, not CI
- `docs/architecture.md` (chapter 1) already defines the 4-agent rationale this chapter's Curriculum Planner implements — don't re-justify it here
- `learning/chapterN.md` format: short Q/A flashcards (see `learning/chapter1.md` for the established style)

---

### Task 1: Scaffold the project with uv

**Files:**
- Create: `pyproject.toml`, `.python-version`, `src/learning_accelerator/__init__.py`
- Modify: `README.md` (Python 3.10+ → 3.11+), `.gitignore` (add checkpoint DB path)
- Create: `.env.example`

**Interfaces:**
- Produces: an installable `learning_accelerator` package importable as `from learning_accelerator.X import Y` from any later task or test

- [ ] **Step 1: Initialize the uv package**

```bash
uv init --package --name learning-accelerator --python 3.11 .
```

Expected: creates `pyproject.toml`, `.python-version`, `src/learning_accelerator/__init__.py`. Does not overwrite the existing `README.md` (uv only creates one if absent).

- [ ] **Step 2: Add runtime and dev dependencies**

```bash
uv add langgraph langgraph-checkpoint-sqlite langchain-core langchain-ollama langchain-anthropic langchain-openai pydantic python-dotenv
uv add --dev pytest
```

Expected: `pyproject.toml` gains a `dependencies` list and a `[dependency-groups] dev` entry; `uv.lock` is created/updated.

- [ ] **Step 3: Verify the environment resolves**

Run: `uv sync`
Expected: exits 0, creates/updates `.venv/`.

- [ ] **Step 4: Update README's Python prerequisite**

In `README.md`, change:
```
- Python 3.10+
```
to:
```
- Python 3.11+
```

- [ ] **Step 5: Add checkpoint DB path to .gitignore**

Add this section to `.gitignore` (after the "Environment variables / secrets" block):

```
# LangGraph SQLite checkpoints
.data/
*.sqlite
*.sqlite3
```

- [ ] **Step 6: Create .env.example**

```
# Choose the LLM provider: ollama | anthropic | openai
LLM_PROVIDER=ollama

# --- Ollama (local, no API key needed) ---
OLLAMA_MODEL=gemma4:12b-mlx
OLLAMA_BASE_URL=http://localhost:11434

# --- Anthropic (used when LLM_PROVIDER=anthropic) ---
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-5

# --- OpenAI (used when LLM_PROVIDER=openai) ---
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Path to the SQLite checkpoint DB (auto-created on first run)
CHECKPOINT_DB_PATH=.data/checkpoints.sqlite
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version src/learning_accelerator/__init__.py README.md .gitignore .env.example
git commit -m "Scaffold learning_accelerator package with uv"
```

---

### Task 2: Shared state schema (`graph/state.py`)

**Files:**
- Create: `src/learning_accelerator/graph/__init__.py`
- Create: `src/learning_accelerator/graph/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: nothing
- Produces: `AgentState` (TypedDict), `Topic`/`StudyRoadmap`/`QuizResult`
  (used by `curriculum_planner.py` in Task 4, and by chapters 3-5's nodes),
  `initial_state(goal: str, session_id: str, study_materials_path: str = "") -> AgentState`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_state.py`:

```python
import pytest
from pydantic import ValidationError

from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state


def test_initial_state_defaults():
    state = initial_state(goal="Learn LangGraph", session_id="abc123")

    assert state["goal"] == "Learn LangGraph"
    assert state["session_id"] == "abc123"
    assert state["messages"] == []
    assert state["roadmap"] is None
    assert state["approved"] is False
    assert state["current_topic_index"] == 0
    assert state["quiz_results"] == []
    assert state["weak_areas"] == []
    assert state["study_materials_path"] == ""
    assert state["error"] is None


def test_initial_state_with_study_materials_path():
    state = initial_state(goal="x", session_id="y", study_materials_path="/tmp/materials")
    assert state["study_materials_path"] == "/tmp/materials"


def test_topic_defaults():
    topic = Topic(title="Intro", description="Basics", estimated_minutes=30)
    assert topic.status == "pending"
    assert topic.prerequisites == []


def test_topic_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Topic(description="Basics", estimated_minutes=30)


def test_study_roadmap_defaults_weekly_hours():
    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=4,
        topics=[Topic(title="Intro", description="Basics", estimated_minutes=30)],
    )
    assert roadmap.weekly_hours == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.graph.state'`

- [ ] **Step 3: Create `src/learning_accelerator/graph/__init__.py`**

```python
```

(empty file — marks `graph` as a package)

- [ ] **Step 4: Write `src/learning_accelerator/graph/state.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Topic(BaseModel):
    title: str
    description: str
    estimated_minutes: int
    prerequisites: list[str] = Field(default_factory=list)
    status: str = "pending"


class StudyRoadmap(BaseModel):
    goal: str
    total_weeks: int
    topics: list[Topic]
    weekly_hours: int = 5


@dataclass
class QuizResult:
    topic: str
    score: float
    passed: bool


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    goal: str
    roadmap: Optional[StudyRoadmap]
    approved: bool
    current_topic_index: int
    quiz_results: list[QuizResult]
    weak_areas: list[str]
    study_materials_path: str
    error: Optional[str]


def initial_state(
    goal: str, session_id: str, study_materials_path: str = ""
) -> AgentState:
    return AgentState(
        messages=[],
        session_id=session_id,
        goal=goal,
        roadmap=None,
        approved=False,
        current_topic_index=0,
        quiz_results=[],
        weak_areas=[],
        study_materials_path=study_materials_path,
        error=None,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/learning_accelerator/graph/__init__.py src/learning_accelerator/graph/state.py tests/test_state.py
git commit -m "Add shared AgentState schema"
```

---

### Task 3: Provider-agnostic LLM config (`config.py`)

**Files:**
- Create: `src/learning_accelerator/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `get_chat_model(temperature: float = 0.1) -> BaseChatModel`, used
  by `curriculum_planner.py` in Task 4 and every later chapter's agent nodes

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.config'`

- [ ] **Step 3: Write `src/learning_accelerator/config.py`**

```python
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}' — expected 'ollama', 'anthropic', or 'openai'"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 passed. If the Anthropic/OpenAI tests fail because the
constructor validates the key format rather than just presence, adjust the
fake key value in the test (not the implementation) until construction
succeeds without a network call.

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/config.py tests/test_config.py
git commit -m "Add provider-agnostic LLM config switch"
```

---

### Task 4: Curriculum Planner node

**Files:**
- Create: `src/learning_accelerator/agents/__init__.py`
- Create: `src/learning_accelerator/agents/curriculum_planner.py`

**Interfaces:**
- Consumes: `get_chat_model` (Task 3), `AgentState`/`StudyRoadmap` (Task 2)
- Produces: `curriculum_planner_node(state: AgentState) -> dict`, wired into
  the graph in Task 5

No unit test here — this node makes a live LLM call, which the spec's
testing approach excludes from pytest. It's verified for real in Task 6's
manual demo run.

- [ ] **Step 1: Create `src/learning_accelerator/agents/__init__.py`**

```python
```

(empty file)

- [ ] **Step 2: Write `src/learning_accelerator/agents/curriculum_planner.py`**

```python
from __future__ import annotations

from langchain_core.messages import AIMessage

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import AgentState, StudyRoadmap

SYSTEM_PROMPT = """You are a curriculum planner. Given a learning goal, \
produce a study roadmap.

Rules:
- Produce 4 to 6 topics, ordered from foundational to advanced.
- Every topic's "prerequisites" must exactly match an earlier topic's title.
- Every topic's "status" is always "pending".
"""


def curriculum_planner_node(state: AgentState) -> dict:
    llm = get_chat_model(temperature=0.1).with_structured_output(StudyRoadmap)

    roadmap = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Learning goal: {state['goal']}"},
        ]
    )

    summary = AIMessage(
        content=(
            f"Planned {len(roadmap.topics)} topics over "
            f"{roadmap.total_weeks} weeks."
        )
    )

    return {
        "roadmap": roadmap,
        "messages": [summary],
        "error": None,
    }
```

- [ ] **Step 3: Commit**

```bash
git add src/learning_accelerator/agents/__init__.py src/learning_accelerator/agents/curriculum_planner.py
git commit -m "Add Curriculum Planner node"
```

---

### Task 5: Graph wiring with SQLite checkpointing

**Files:**
- Create: `src/learning_accelerator/graph/workflow.py`

**Interfaces:**
- Consumes: `curriculum_planner_node` (Task 4), `AgentState` (Task 2)
- Produces: `build_graph(db_path: str = ...) -> CompiledStateGraph`, `graph`
  (module-level compiled instance) — chapters 3-5 will add more
  `builder.add_node(...)`/`add_edge(...)` calls to this same file rather
  than replacing it

- [ ] **Step 1: Write `src/learning_accelerator/graph/workflow.py`**

```python
from __future__ import annotations

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from learning_accelerator.agents.curriculum_planner import curriculum_planner_node
from learning_accelerator.graph.state import AgentState

DEFAULT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", ".data/checkpoints.sqlite")


def build_graph(db_path: str = DEFAULT_DB_PATH):
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # check_same_thread=False: LangGraph runs node functions and checkpoint
    # writes on different threads, so the connection can't be thread-bound.
    # No context manager: this connection must survive the whole process,
    # not just this function call.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(AgentState)
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
```

- [ ] **Step 2: Commit**

```bash
git add src/learning_accelerator/graph/workflow.py
git commit -m "Wire curriculum_planner into graph with SQLite checkpointing"
```

---

### Task 6: Manual demo script + real verification run

**Files:**
- Create: `scripts/demo_chapter2.py`

**Interfaces:**
- Consumes: `graph`, `initial_state` from Tasks 2 and 5
- Produces: nothing consumed by later tasks — this is the chapter's
  manual-run verification artifact

- [ ] **Step 1: Write `scripts/demo_chapter2.py`**

```python
"""Manual run: invoke the chapter 2 graph (Curriculum Planner only).

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly.
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
    print(f"Roadmap: {result['roadmap']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real against local Ollama**

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b-mlx uv run python scripts/demo_chapter2.py
```

Expected: prints a session ID and a `StudyRoadmap` with 4-6 topics. If
`with_structured_output` fails because `gemma4:12b-mlx` doesn't support
tool-calling-based structured output reliably, retry with `gemma4:12b` or
`qwen3.5:2b` (also available locally per `ollama list`) and note in the
chapter 2 flashcards which model(s) actually worked.

- [ ] **Step 3: Run the full pytest suite once more**

Run: `uv run pytest -v`
Expected: all tests from Tasks 2 and 3 still pass (this step only adds a
script, no new pytest tests).

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_chapter2.py
git commit -m "Add chapter 2 manual demo script"
```

---

### Task 7: Chapter 2 learning flashcards

**Files:**
- Create: `learning/chapter2.md`

- [ ] **Step 1: Write `learning/chapter2.md`**

```markdown
## Chapter 2: Stateful Orchestration with LangGraph

**Q: What reducer does the `messages` field in `AgentState` use, and why?**
A: `Annotated[list[BaseMessage], add_messages]` — appends new messages
instead of overwriting the list, so conversation history accumulates
across every agent in the graph.

**Q: How do other `AgentState` fields (e.g. `roadmap`, `approved`) update?**
A: Last-write-wins — a node's returned value for that key replaces the
previous one (no reducer).

**Q: What does a node function return — the full state or a partial update?**
A: A partial dict of only the keys it changed. LangGraph merges it into
the existing state using each field's reducer (or last-write-wins).

**Q: Why is `SqliteSaver` constructed from a raw `sqlite3.Connection`
instead of `SqliteSaver.from_conn_string(...)` as a context manager?**
A: LangGraph runs node functions and checkpoint writes on different
threads (`check_same_thread=False`), and the connection must stay open
for the whole process — a `with` block would close it too early.

**Q: Why are `Topic`/`StudyRoadmap` Pydantic models here instead of the
source article's plain dataclasses?**
A: `llm.with_structured_output(StudyRoadmap)` needs a Pydantic model (or
JSON schema) to work identically across Ollama/Anthropic/OpenAI — this
repo's provider-agnostic goal the article doesn't need to solve.

**Q: What temperature does the Curriculum Planner use, and why?**
A: `0.1` — planning wants deterministic, consistent structured output,
not creative variation.

**Q: What does chapter 2's graph actually wire up?**
A: Only `START → curriculum_planner → END`. The other four agents
(explainer, quiz generator, progress coach, human approval) don't exist
yet — they're added in chapters 3-5.

**Q: What local Ollama model was used to verify this chapter, and did it
need adjusting?**
A: See the commit history / demo script run output for this repo — record
whichever of `gemma4:12b-mlx`, `gemma4:12b`, or `qwen3.5:2b` actually
produced a valid structured `StudyRoadmap`.
```

Before committing, fill in the last flashcard's answer with what Task 6
actually observed (which model worked).

- [ ] **Step 2: Commit**

```bash
git add learning/chapter2.md
git commit -m "Add chapter 2 learning flashcards"
```

---

### Task 8: Open the chapter 2 PR

**Files:** none (branch/PR operation only)

- [ ] **Step 1: Confirm all tests pass and demo ran successfully**

Run: `uv run pytest -v`
Expected: all tests pass (from Tasks 2 and 3).

- [ ] **Step 2: Push the branch**

**Before running this step, ask the user to confirm.**

```bash
git push -u origin agent/chapter-2
```

- [ ] **Step 3: Open the PR**

**Before running this step, ask the user to confirm.**

```bash
gh pr create --title "Chapter 2: Stateful Orchestration with LangGraph" --body "$(cat <<'EOF'
## Summary
- Scaffold the project with uv (src-layout `learning_accelerator` package)
- Add shared `AgentState` schema (`graph/state.py`) with `Topic`/`StudyRoadmap`
  as Pydantic models for cross-provider structured output
- Add provider-agnostic `config.get_chat_model()` (ollama/anthropic/openai)
- Add the Curriculum Planner node and wire it into a graph with SQLite
  checkpointing (`START → curriculum_planner → END`)
- Add a manual demo script, verified against local Ollama
- Add learning/chapter2.md flashcards

## Test plan
- [ ] `uv sync && uv run pytest -v` — all tests pass
- [ ] `uv run python scripts/demo_chapter2.py` — prints a session ID and a
      StudyRoadmap with 4-6 topics

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report the PR URL to the user**
