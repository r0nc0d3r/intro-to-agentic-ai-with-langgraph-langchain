# Runtime Performance Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant work in the Learning Accelerator's hot path for a single-user local study session, via three small, independent changes.

**Architecture:** No structural changes — each task is a small, self-contained edit to existing modules (`config.py`, `graph/state.py`, `graph/workflow.py`) plus their call sites, verified by tests before and after. Each task is its own commit and can ship as its own PR.

**Tech Stack:** Python 3.11, `functools.lru_cache` (stdlib, no new dependency), `pytest` (existing dev dependency), `uv run pytest -q` to execute tests.

## Global Constraints

- Scope is single-user local session runtime performance only — no dependency changes, no multi-user/concurrency changes, no changes to A2A probe behavior or `state["messages"]` growth (all explicitly out of scope per `docs/superpowers/specs/2026-07-29-runtime-performance-pass-design.md`).
- Each of the 3 tasks below is committed separately (own commit, own PR) — do not squash them together.
- The full test suite must stay green after every task: `uv run pytest -q` currently reports `123 passed, 12 deselected` — that count should only grow (by the new tests each task adds), never shrink or regress.
- Commit messages follow this repo's `CONTRIBUTING.md` style: imperative summary line (~50-72 chars), blank line, body explaining *why* when non-obvious.
- Source spec: `docs/superpowers/specs/2026-07-29-runtime-performance-pass-design.md` — refer back to it for the full rationale behind each change; this plan contains everything needed to implement it, but the spec has the "why" in more depth.

---

### Task 1: Cache LLM client construction

**Files:**
- Modify: `src/learning_accelerator/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `get_chat_model(temperature: float = 0.1) -> BaseChatModel` — same signature as today, now memoized via `functools.lru_cache`. Callers in `curriculum_planner.py`, `explainer.py`, `quiz_generator.py`, `progress_coach.py`, and `evaluation/judge_model.py` are unaffected and need no changes.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_config.py`:

```python
def test_get_chat_model_caches_same_temperature(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    model1 = get_chat_model(0.2)
    model2 = get_chat_model(0.2)
    assert model1 is model2


def test_get_chat_model_different_temperature_not_cached(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    model1 = get_chat_model(0.2)
    model2 = get_chat_model(0.3)
    assert model1 is not model2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: the two new tests FAIL (`assert model1 is model2` fails — each call currently constructs a fresh, distinct object).

- [ ] **Step 3: Implement caching, and keep the existing tests correct under it**

The five existing tests in `tests/test_config.py` all call `get_chat_model()` with the same default `temperature=0.1`, each expecting a *different* result based on `monkeypatch`-ed env vars. Once caching is added, they'd otherwise all collide on the same cache entry and only the first would ever construct a real model. Add an autouse fixture that clears the cache before and after every test in this file, at the top of `tests/test_config.py` (after the existing imports):

```python
import pytest

from learning_accelerator.config import get_chat_model


@pytest.fixture(autouse=True)
def _clear_chat_model_cache():
    get_chat_model.cache_clear()
    yield
    get_chat_model.cache_clear()
```

(This replaces the current bare `import pytest` / `from learning_accelerator.config import get_chat_model` lines at the top of the file — keep those two imports, just add the fixture right after them.)

In `src/learning_accelerator/config.py`, add the `lru_cache` import and decorator:

```python
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


@lru_cache(maxsize=None)
def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    ...  # rest of the function body is unchanged
```

Only the `import os` → add `from functools import lru_cache` line and the `@lru_cache(maxsize=None)` decorator are new. The function body (the `if provider == ...` chain and the final `raise ValueError`) stays exactly as-is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 7 tests PASS (the original 5 + the 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
Cache LLM client construction in get_chat_model

get_chat_model() built a new provider client on every call, but every
node/agent only ever uses one of ~4 fixed temperatures per process —
memoizing on (temperature) avoids reconstructing the same client
repeatedly within a session.
EOF
)"
```

---

### Task 2: Extract duplicated "last AIMessage" scan

**Files:**
- Modify: `src/learning_accelerator/graph/state.py`
- Modify: `src/learning_accelerator/agents/quiz_generator.py`
- Modify: `src/learning_accelerator/agents/progress_coach.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `get_last_explanation(state: AgentState) -> str` in `graph/state.py` — returns the most recent non-tool-call `AIMessage`'s content, or `""` if none exists. Used by `quiz_generator.py::quiz_generator_node` and `progress_coach.py::progress_coach_node`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_state.py`. First, extend the existing imports at the top of the file:

```python
from langchain_core.messages import AIMessage, HumanMessage

from learning_accelerator.graph.state import (
    GradedAnswer,
    QuizResult,
    StudyRoadmap,
    Topic,
    get_current_topic,
    get_last_explanation,
    initial_state,
    session_is_complete,
)
```

Then add these three tests anywhere in the file:

```python
def test_get_last_explanation_returns_empty_when_no_messages():
    state = initial_state(goal="g", session_id="s")
    assert get_last_explanation(state) == ""


def test_get_last_explanation_skips_tool_call_messages():
    state = initial_state(goal="g", session_id="s")
    state["messages"] = [
        AIMessage(content="explanation text"),
        AIMessage(
            content="",
            tool_calls=[{"name": "tool_read_file", "args": {}, "id": "1"}],
        ),
    ]
    assert get_last_explanation(state) == "explanation text"


def test_get_last_explanation_returns_most_recent():
    state = initial_state(goal="g", session_id="s")
    state["messages"] = [
        AIMessage(content="first explanation"),
        HumanMessage(content="ignored, not an AIMessage"),
        AIMessage(content="second explanation"),
    ]
    assert get_last_explanation(state) == "second explanation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_last_explanation'`.

- [ ] **Step 3: Implement the helper**

In `src/learning_accelerator/graph/state.py`, change the messages import on line 6 from:

```python
from langchain_core.messages import BaseMessage
```

to:

```python
from langchain_core.messages import AIMessage, BaseMessage
```

Then add this function after `get_current_topic` (which ends around line 84):

```python
def get_last_explanation(state: AgentState) -> str:
    """Return the most recent non-tool-call AIMessage's content, or ""."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: all tests in the file PASS, including the 3 new ones.

- [ ] **Step 5: Use the helper in quiz_generator.py**

In `src/learning_accelerator/agents/quiz_generator.py`:

Change the import block (currently):
```python
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import (
    PASS_THRESHOLD,
    AgentState,
    QuizResult,
    get_current_topic,
)
```
to (drop the now-unused `AIMessage` import, add `get_last_explanation`):
```python
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import (
    PASS_THRESHOLD,
    AgentState,
    QuizResult,
    get_current_topic,
    get_last_explanation,
)
```

Then in `quiz_generator_node`, replace:
```python
    explanation = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            explanation = msg.content
            break
```
with:
```python
    explanation = get_last_explanation(state)
```

- [ ] **Step 6: Use the helper in progress_coach.py**

In `src/learning_accelerator/agents/progress_coach.py`, add `get_last_explanation` to the existing `graph.state` import (line 10):

Change:
```python
from learning_accelerator.graph.state import AgentState, PASS_THRESHOLD, session_is_complete
```
to:
```python
from learning_accelerator.graph.state import (
    AgentState,
    PASS_THRESHOLD,
    get_last_explanation,
    session_is_complete,
)
```
(Note: `AIMessage` stays imported in this file — it's still used later to build the coaching message.)

Then in `progress_coach_node`, replace:
```python
        explanation = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                explanation = msg.content
                break
```
with:
```python
        explanation = get_last_explanation(state)
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, same or higher count than before this task (no regressions in `test_progress_coach.py`, `test_workflow.py`, or anywhere else that exercises these two node functions).

- [ ] **Step 8: Commit**

```bash
git add src/learning_accelerator/graph/state.py \
        src/learning_accelerator/agents/quiz_generator.py \
        src/learning_accelerator/agents/progress_coach.py \
        tests/test_state.py
git commit -m "$(cat <<'EOF'
Extract get_last_explanation helper, dedupe from two agents

quiz_generator_node and progress_coach_node each had an identical
inline loop for finding the last non-tool-call AIMessage. Moved to
graph/state.py alongside the other shared state helpers
(get_current_topic, session_is_complete).
EOF
)"
```

---

### Task 3: Lazy graph construction

**Files:**
- Modify: `src/learning_accelerator/graph/workflow.py`
- Modify: `main.py`
- Modify: `scripts/demo_chapter2.py`
- Modify: `scripts/demo_chapter3.py`
- Modify: `scripts/demo_chapter4.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Produces: `get_default_graph()` in `graph/workflow.py` — a zero-argument, `lru_cache(maxsize=1)`-memoized function returning the same compiled graph `build_graph()` would with all defaults. Replaces the old module-level `graph = build_graph()` singleton.
- Consumes: `build_graph()` (already exists, unchanged signature).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workflow.py`. First extend the imports at the top of the file:

```python
import subprocess
import sys

from learning_accelerator.graph.workflow import build_graph
```

Then add these two tests:

```python
def test_importing_workflow_does_not_create_checkpoint_db(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import learning_accelerator.graph.workflow"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / ".data").exists()


def test_calling_get_default_graph_creates_checkpoint_db(tmp_path):
    script = (
        "import learning_accelerator.graph.workflow as w\n"
        "w.get_default_graph()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".data" / "checkpoints.sqlite").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: `test_importing_workflow_does_not_create_checkpoint_db` FAILS (today's code creates the DB on import, since `graph = build_graph()` runs at module load). `test_calling_get_default_graph_creates_checkpoint_db` FAILS (`AttributeError: module '...workflow' has no attribute 'get_default_graph'`, surfaced as a non-zero subprocess return code).

- [ ] **Step 3: Implement lazy construction**

In `src/learning_accelerator/graph/workflow.py`, add the `lru_cache` import (it already imports from `os` and `sqlite3`):

Change:
```python
from __future__ import annotations

import os
import sqlite3
```
to:
```python
from __future__ import annotations

import os
import sqlite3
from functools import lru_cache
```

Then replace the last line of the file:
```python
graph = build_graph()
```
with:
```python
@lru_cache(maxsize=1)
def get_default_graph():
    return build_graph()
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: all 4 tests in the file PASS (the original 2 `build_graph` tests + the 2 new ones).

- [ ] **Step 5: Update main.py**

In `main.py`, change the import (line 19):
```python
from learning_accelerator.graph.workflow import graph
```
to:
```python
from learning_accelerator.graph.workflow import get_default_graph
```

Then change both call sites. First, in `run_session` (currently):
```python
    try:
        result = graph.invoke(state, config=config)
```
to:
```python
    try:
        result = get_default_graph().invoke(state, config=config)
```

Second, later in the same function:
```python
        result = graph.invoke(Command(resume=user_input), config=config)
```
to:
```python
        result = get_default_graph().invoke(Command(resume=user_input), config=config)
```

- [ ] **Step 6: Update scripts/demo_chapter2.py**

Change:
```python
from learning_accelerator.graph.workflow import graph
```
to:
```python
from learning_accelerator.graph.workflow import get_default_graph
```
And change:
```python
    result = graph.invoke(state, config=config)
```
to:
```python
    result = get_default_graph().invoke(state, config=config)
```

- [ ] **Step 7: Update scripts/demo_chapter3.py**

Same two edits as Step 6 (identical import line and identical `graph.invoke(state, config=config)` call in this file).

- [ ] **Step 8: Update scripts/demo_chapter4.py**

Change:
```python
from learning_accelerator.graph.workflow import graph
```
to:
```python
from learning_accelerator.graph.workflow import get_default_graph
```
And change:
```python
    result = graph.invoke(state, config=config)
```
to:
```python
    result = get_default_graph().invoke(state, config=config)
```

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, same or higher count than before this task.

- [ ] **Step 10: Sanity-check the updated entry points still import cleanly**

Run: `uv run python -c "import ast; [ast.parse(open(f).read(), f) for f in ['main.py', 'scripts/demo_chapter2.py', 'scripts/demo_chapter3.py', 'scripts/demo_chapter4.py']]"`
Expected: no output, exit code 0 (confirms no syntax errors introduced by the edits).

Run: `uv run python -m py_compile main.py scripts/demo_chapter2.py scripts/demo_chapter3.py scripts/demo_chapter4.py`
Expected: no output, exit code 0.

- [ ] **Step 11: Commit**

```bash
git add src/learning_accelerator/graph/workflow.py main.py \
        scripts/demo_chapter2.py scripts/demo_chapter3.py scripts/demo_chapter4.py \
        tests/test_workflow.py
git commit -m "$(cat <<'EOF'
Make default graph construction lazy

graph = build_graph() ran at import time, opening a SQLite checkpoint
connection just from importing the module — even if the caller never
invoked the graph. get_default_graph() (lru_cache-memoized) defers
that until first real use. Updates the 4 call sites that relied on
the old module-level singleton (streamlit_app.py and tests already
call build_graph() directly and are unaffected).
EOF
)"
```
