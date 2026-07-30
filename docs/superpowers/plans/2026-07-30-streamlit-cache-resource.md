# Streamlit `@st.cache_resource` Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `streamlit_app.py` from rebuilding `ui_graph` (and its SQLite connection) on every Streamlit rerun, by wrapping its construction in `@st.cache_resource`.

**Architecture:** One task, two files. `streamlit_app.py` gets its graph construction wrapped in a cached `get_ui_graph()` function; `tests/test_streamlit_app.py` gets a one-line companion fix (clearing that cache before each test) that is required for the first change to be safe — without it, cached state leaks across tests. Both changes ship in one commit since the second is a correctness requirement of the first, not an optional extra.

**Tech Stack:** Python 3.11, Streamlit's `@st.cache_resource` decorator (already a project dependency, no new package), `streamlit.testing.v1.AppTest` (already used by the existing test suite).

## Global Constraints

- No other line in `streamlit_app.py` changes besides the graph-construction block — every `ui_graph.invoke(...)`/`.update_state(...)`/`.get_state(...)` call site elsewhere in the file stays exactly as-is.
- The test-isolation fix goes inside the existing `_mock_build_graph` helper in `tests/test_streamlit_app.py` (confirmed via grep: all 17 existing test functions already call it) — do not add a new fixture or touch any individual test function.
- All 17 existing tests in `tests/test_streamlit_app.py` must stay green; no new test cases are needed (see spec's Testing section) — the existing suite is both the fix's regression guard and its verification.
- Full repo test suite must stay green: run `uv run pytest -q` before committing. Baseline going into this task depends on whether PR #20 (node-function tests) has merged yet — run the suite yourself first to establish the current baseline before making changes, rather than assuming a specific number.
- Commit message follows this repo's `CONTRIBUTING.md` style: imperative summary line (~50-72 chars), blank line, body explaining *why* when non-obvious.
- Source spec: `docs/superpowers/specs/2026-07-30-streamlit-cache-resource-design.md` — refer back to it for full rationale, including the empirical verification of the cache-leakage behavior this task's test fix addresses.

---

### Task 1: Cache `ui_graph` construction and fix test isolation

**Files:**
- Modify: `streamlit_app.py:46-49`
- Modify: `tests/test_streamlit_app.py:11-30`

**Interfaces:**
- Produces: `get_ui_graph() -> CompiledStateGraph` (a new, `@st.cache_resource`-decorated, zero-argument function in `streamlit_app.py`). Nothing outside this file consumes it; `ui_graph` (the module-level name every other line in the file already uses) is reassigned to `get_ui_graph()`'s return value in the same place the old direct `build_graph(...)` call used to be.

- [ ] **Step 1: Establish the current baseline**

Run: `uv run pytest -q`
Note the reported pass count (e.g. "137 passed, 12 deselected") — you'll compare against this exact number in Step 5, not a number from this plan document (which may be stale if other work has landed on `main` since this plan was written).

- [ ] **Step 2: Apply the core fix to `streamlit_app.py`**

Find this block (currently lines 46-49):

```python
ui_graph = build_graph(
    db_path=".data/checkpoints_ui.sqlite",
    interrupt_before=["quiz_generator"],
)
```

Replace it with:

```python
@st.cache_resource
def get_ui_graph():
    return build_graph(
        db_path=".data/checkpoints_ui.sqlite",
        interrupt_before=["quiz_generator"],
    )


ui_graph = get_ui_graph()
```

Do not change anything else in the file — every later `ui_graph.invoke(...)`, `ui_graph.update_state(...)`, `ui_graph.get_state(...)` call (in `start_session`, `approve_roadmap`, and `advance_after_quiz`) stays exactly as it is today.

- [ ] **Step 3: Apply the test-isolation fix to `tests/test_streamlit_app.py`**

The file currently starts with (lines 1-31):

```python
"""UI-layer tests for streamlit_app.py using Streamlit's AppTest harness.

These tests never touch a real LLM or LangGraph checkpoint: ``build_graph``
is replaced with a MagicMock so ``ui_graph.invoke`` / ``.update_state`` /
``.get_state`` return canned results, and the quiz agent functions
(``generate_questions`` / ``grade_answer``) are stubbed too. The goal is to
cover the Streamlit screen-routing and session-state logic itself — the
graph/agent behavior already has its own test coverage elsewhere.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from learning_accelerator.agents.quiz_generator import GradeResult, QuizQuestion
from learning_accelerator.graph.state import QuizResult, StudyRoadmap, Topic

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _mock_build_graph(monkeypatch, mock_graph: MagicMock) -> None:
    monkeypatch.setattr(
        "learning_accelerator.graph.workflow.build_graph",
        lambda **kwargs: mock_graph,
    )
```

Make two changes:

1. Add `import streamlit as st` to the import block. Change:
   ```python
   from langchain_core.messages import AIMessage
   from streamlit.testing.v1 import AppTest
   ```
   to:
   ```python
   import streamlit as st
   from langchain_core.messages import AIMessage
   from streamlit.testing.v1 import AppTest
   ```

2. Add a cache-clear as the first line of `_mock_build_graph`'s body. Change:
   ```python
   def _mock_build_graph(monkeypatch, mock_graph: MagicMock) -> None:
       monkeypatch.setattr(
           "learning_accelerator.graph.workflow.build_graph",
           lambda **kwargs: mock_graph,
       )
   ```
   to:
   ```python
   def _mock_build_graph(monkeypatch, mock_graph: MagicMock) -> None:
       st.cache_resource.clear()
       monkeypatch.setattr(
           "learning_accelerator.graph.workflow.build_graph",
           lambda **kwargs: mock_graph,
       )
   ```

No other line in the file changes — all 17 test functions already call `_mock_build_graph`, so this one edit covers the whole file.

- [ ] **Step 4: Run the Streamlit test file to verify isolation holds**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: all 17 tests PASS. This is the direct proof the fix works — if cache isolation were broken, tests would start failing or asserting against the wrong mock's data (e.g. a later test seeing an earlier test's roadmap/quiz data), since each test's mock returns distinct values.

- [ ] **Step 5: Run the full test suite to verify no regressions**

Run: `uv run pytest -q`
Expected: same pass count as Step 1's baseline (this change adds no new tests — it fixes an existing file's behavior under a new caching layer), 0 failures.

- [ ] **Step 6: Manual sanity check (optional but recommended for a UI change)**

If you have Ollama running locally, you can confirm the fix visually:
```bash
uv run streamlit run streamlit_app.py
```
Interact with the app (submit a goal, approve the roadmap) and confirm it behaves identically to before — the fix is purely about avoiding redundant reconstruction, not changing any visible behavior. This step has no pass/fail assertion; skip it if Ollama isn't available in your environment, since Steps 4-5 already provide full automated verification.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "$(cat <<'EOF'
Cache ui_graph construction across Streamlit reruns

streamlit_app.py rebuilt ui_graph (opening a new SQLite connection and
recompiling the StateGraph) on every widget interaction, since
Streamlit re-executes the whole script top-to-bottom on each rerun.
Wrapped construction in @st.cache_resource so it happens once per
process instead. Required a companion fix in the test suite: verified
empirically that the new cache leaks across separate
AppTest.from_file() calls in the same pytest process unless cleared,
so _mock_build_graph now clears it before each test.
EOF
)"
```
