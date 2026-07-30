# Node Function Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fast-suite unit tests for the 4 LangGraph node functions (`curriculum_planner_node`, `explainer_node`, `quiz_generator_node`, `progress_coach_node`) that currently have zero coverage of their own orchestration logic.

**Architecture:** Four independent tasks, one per node function, each producing its own commit. Each test mocks at the nearest already-tested boundary (either `get_chat_model` directly, or an already-tested collaborator function) rather than hitting a real LLM — see each task's exact mocking code below.

**Tech Stack:** Python 3.11, `pytest`, `unittest.mock.MagicMock` (already the established mocking library in this suite — see `tests/test_judge_model.py`), `uv run pytest` to execute.

## Global Constraints

- Coverage depth is "happy path + key branches" per node, not exhaustive — exactly the tests listed in each task below, nothing more.
- Mock boundary: `curriculum_planner_node` and `explainer_node` mock `get_chat_model` directly (no lower boundary exists). `quiz_generator_node` mocks `run_quiz`. `progress_coach_node` mocks `get_coaching_message` and `try_study_buddy_assistance`. Do not mock deeper (e.g. don't mock `get_chat_model` inside `quiz_generator.py` for the node test — `run_quiz` is the boundary).
- `explainer_node`'s tool calls are NOT mocked — they run for real against `study_materials/sample_notes/` (a fast, local, deterministic file read).
- Full test suite must stay green after every task: run `uv run pytest -q` before each commit. Current baseline (verify this yourself before Task 1): should be 130 passed, 12 deselected — that count should only grow, never shrink.
- Commit messages follow this repo's `CONTRIBUTING.md` style: imperative summary line (~50-72 chars), blank line, body explaining *why* when non-obvious.
- Source spec: `docs/superpowers/specs/2026-07-30-node-test-coverage-design.md` — refer back to it for full rationale; this plan contains everything needed to implement it.

---

### Task 1: `curriculum_planner_node` test

**Files:**
- Create: `tests/test_curriculum_planner.py`

**Interfaces:**
- Consumes: `curriculum_planner_node(state: AgentState) -> dict` (unchanged, from `src/learning_accelerator/agents/curriculum_planner.py`), `StudyRoadmap`/`Topic`/`initial_state` from `src/learning_accelerator/graph/state.py` (all unchanged, already exist).

- [ ] **Step 1: Write the failing test**

Create `tests/test_curriculum_planner.py` with this exact content:

```python
from unittest.mock import MagicMock

from learning_accelerator.agents import curriculum_planner
from learning_accelerator.agents.curriculum_planner import curriculum_planner_node
from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state


def test_curriculum_planner_node_returns_roadmap_and_summary_message(monkeypatch):
    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=4,
        topics=[
            Topic(title="Intro", description="Basics", estimated_minutes=30),
            Topic(title="State", description="State management", estimated_minutes=45),
        ],
    )
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = roadmap
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm
    monkeypatch.setattr(
        curriculum_planner, "get_chat_model", MagicMock(return_value=fake_llm)
    )

    state = initial_state(goal="Learn LangGraph", session_id="s1")
    result = curriculum_planner_node(state)

    assert result["roadmap"] is roadmap
    assert result["error"] is None
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Planned 2 topics over 4 weeks."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_curriculum_planner.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` if the file has a typo, or (if imports are fine) an `AssertionError`/`AttributeError` because `get_chat_model` isn't yet monkeypatched correctly — actually, since this test only *adds* a new test file against unchanged production code, it should PASS immediately if written correctly. If it fails, the failure will point at a real mismatch between this plan's assumptions and the actual code — read the failure carefully before proceeding, do not assume the test is simply "not implemented yet" as with new-feature TDD.

- [ ] **Step 3: Confirm the test passes as-is**

Run: `uv run pytest tests/test_curriculum_planner.py -v`
Expected: PASS (1 passed). This task adds test coverage for existing, unchanged behavior — there is no implementation step, since `curriculum_planner_node` already exists and works. If the test fails, investigate whether the mock chain doesn't match `curriculum_planner.py`'s actual call shape (`get_chat_model(temperature=0.1).with_structured_output(StudyRoadmap)`) before changing anything else.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, count increased by exactly 1 over the baseline.

- [ ] **Step 5: Commit**

```bash
git add tests/test_curriculum_planner.py
git commit -m "$(cat <<'EOF'
Add unit test for curriculum_planner_node

curriculum_planner_node had zero fast-suite coverage of its own
orchestration (state read/write shape) despite being a core graph
node — only exercised previously via the eval-marked, real-LLM test
suite excluded from the default run.
EOF
)"
```

---

### Task 2: `explainer_node` tests

**Files:**
- Create: `tests/test_explainer.py`

**Interfaces:**
- Consumes: `explainer_node(state: AgentState) -> dict` and `MAX_ITERATIONS` (both from `src/learning_accelerator/agents/explainer.py`, unchanged), `StudyRoadmap`/`Topic`/`initial_state` from `graph/state.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_explainer.py` with this exact content:

```python
from unittest.mock import MagicMock

from learning_accelerator.agents import explainer
from learning_accelerator.agents.explainer import MAX_ITERATIONS, explainer_node
from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state


def _make_state_with_topic():
    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=4,
        topics=[Topic(title="Intro", description="Basics", estimated_minutes=30)],
    )
    state = initial_state(goal="Learn LangGraph", session_id="s1")
    state["roadmap"] = roadmap
    return state


def test_explainer_node_returns_final_response_after_one_tool_round_trip(monkeypatch):
    state = _make_state_with_topic()

    tool_call_response = MagicMock()
    tool_call_response.tool_calls = [
        {"name": "tool_list_files", "args": {}, "id": "call_1"}
    ]

    final_response = MagicMock()
    final_response.tool_calls = []
    final_response.content = "LangGraph basics explained."

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [tool_call_response, final_response]
    fake_chat_model = MagicMock()
    fake_chat_model.bind_tools.return_value = fake_llm
    monkeypatch.setattr(
        explainer, "get_chat_model", MagicMock(return_value=fake_chat_model)
    )

    result = explainer_node(state)

    assert result == {"messages": [final_response], "error": None}
    assert fake_llm.invoke.call_count == 2


def test_explainer_node_returns_error_when_max_iterations_exceeded(monkeypatch):
    state = _make_state_with_topic()

    looping_response = MagicMock()
    looping_response.tool_calls = [
        {"name": "tool_list_files", "args": {}, "id": "call_loop"}
    ]

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = looping_response
    fake_chat_model = MagicMock()
    fake_chat_model.bind_tools.return_value = fake_llm
    monkeypatch.setattr(
        explainer, "get_chat_model", MagicMock(return_value=fake_chat_model)
    )

    result = explainer_node(state)

    assert result == {"error": "explainer exceeded max iterations"}
    assert fake_llm.invoke.call_count == MAX_ITERATIONS
```

Note: `tool_call_response.tool_calls` uses the real `tool_list_files` tool name, so `_execute_tool_call` in `explainer.py` will actually invoke the real `list_study_files()` MCP function against `study_materials/sample_notes/` during the first test — this is intentional (see Global Constraints), not a mistake to fix.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_explainer.py -v`
Expected: as in Task 1 — this targets existing, unchanged behavior, so it should PASS if written correctly. If it fails, read the failure carefully: it likely means the mock chain shape (`get_chat_model(...).bind_tools(...)`) doesn't match `explainer.py`'s actual code, not that a new feature needs implementing.

- [ ] **Step 3: Confirm both tests pass as-is**

Run: `uv run pytest tests/test_explainer.py -v`
Expected: PASS (2 passed).

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, count increased by exactly 2 over Task 1's count.

- [ ] **Step 5: Commit**

```bash
git add tests/test_explainer.py
git commit -m "$(cat <<'EOF'
Add unit tests for explainer_node

Covers the happy path (one real tool round-trip against real sample
notes, then a final non-tool-call response) and the max-iterations
error path — explainer_node's iterative tool-calling loop had zero
fast-suite coverage before this.
EOF
)"
```

---

### Task 3: `quiz_generator_node` tests

**Files:**
- Create: `tests/test_quiz_generator.py`

**Interfaces:**
- Consumes: `quiz_generator_node(state: AgentState) -> dict` and `run_quiz` (both from `src/learning_accelerator/agents/quiz_generator.py`, unchanged), `QuizResult`/`StudyRoadmap`/`Topic`/`initial_state` from `graph/state.py`, `AIMessage` from `langchain_core.messages`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_quiz_generator.py` with this exact content:

```python
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from learning_accelerator.agents import quiz_generator
from learning_accelerator.agents.quiz_generator import quiz_generator_node
from learning_accelerator.graph.state import QuizResult, StudyRoadmap, Topic, initial_state


def _make_state_with_explanation(weak_areas=None):
    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=4,
        topics=[Topic(title="Intro", description="Basics", estimated_minutes=30)],
    )
    state = initial_state(goal="Learn LangGraph", session_id="s1")
    state["roadmap"] = roadmap
    state["messages"] = [AIMessage(content="Here is the explanation of Intro.")]
    if weak_areas is not None:
        state["weak_areas"] = weak_areas
    return state


def test_quiz_generator_node_appends_result_and_merges_weak_areas(monkeypatch):
    state = _make_state_with_explanation(weak_areas=["variables"])

    quiz_result = QuizResult(
        topic="Intro", score=0.8, passed=True, weak_areas=["recursion"]
    )
    mock_run_quiz = MagicMock(return_value=quiz_result)
    monkeypatch.setattr(quiz_generator, "run_quiz", mock_run_quiz)

    result = quiz_generator_node(state)

    assert result["quiz_results"] == [quiz_result]
    assert sorted(result["weak_areas"]) == sorted(["variables", "recursion"])
    assert result["error"] is None
    mock_run_quiz.assert_called_once_with("Intro", "Here is the explanation of Intro.")


def test_quiz_generator_node_deduplicates_weak_areas(monkeypatch):
    state = _make_state_with_explanation(weak_areas=["closures"])

    quiz_result = QuizResult(
        topic="Intro", score=0.4, passed=False, weak_areas=["closures", "recursion"]
    )
    monkeypatch.setattr(quiz_generator, "run_quiz", MagicMock(return_value=quiz_result))

    result = quiz_generator_node(state)

    assert sorted(result["weak_areas"]) == sorted(["closures", "recursion"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quiz_generator.py -v`
Expected: as in prior tasks — targets existing, unchanged behavior, should PASS if written correctly. A failure here means the mocked `run_quiz` call shape doesn't match `quiz_generator_node`'s actual call (`run_quiz(topic.title, explanation)`, positional args) — check `src/learning_accelerator/agents/quiz_generator.py`'s `quiz_generator_node` function before changing the test.

- [ ] **Step 3: Confirm both tests pass as-is**

Run: `uv run pytest tests/test_quiz_generator.py -v`
Expected: PASS (2 passed).

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, count increased by exactly 2 over Task 2's count.

- [ ] **Step 5: Commit**

```bash
git add tests/test_quiz_generator.py
git commit -m "$(cat <<'EOF'
Add unit tests for quiz_generator_node

Covers the happy path (quiz_results appended, weak_areas merged,
run_quiz called with the topic and extracted explanation) and the
weak-area de-duplication branch — quiz_generator_node's own
orchestration had zero fast-suite coverage before this (run_quiz
itself was already tested).
EOF
)"
```

---

### Task 4: `progress_coach_node` tests

**Files:**
- Modify: `tests/test_progress_coach.py` (extend — add imports and 2 new test functions; do not change the existing 6 tests)

**Interfaces:**
- Consumes: `progress_coach_node(state: AgentState) -> dict` and `CoachingMessage` (both from `src/learning_accelerator/agents/progress_coach.py`, unchanged), `QuizResult`/`StudyRoadmap`/`Topic`/`initial_state` from `graph/state.py`, `AIMessage` from `langchain_core.messages`.

- [ ] **Step 1: Write the failing tests**

`tests/test_progress_coach.py` currently starts with:

```python
from learning_accelerator.agents.progress_coach import (
    next_topic_status,
    route_after_coach,
)
from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state
```

Replace those import lines (keep everything below them — the existing 6 test functions — completely unchanged) with:

```python
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from learning_accelerator.agents import progress_coach
from learning_accelerator.agents.progress_coach import (
    CoachingMessage,
    next_topic_status,
    progress_coach_node,
    route_after_coach,
)
from learning_accelerator.graph.state import QuizResult, StudyRoadmap, Topic, initial_state
```

Then append these two new test functions (plus their shared helper) at the end of the file, after the existing `test_route_after_coach_ends_when_topics_exhausted`:

```python
def _make_state_with_quiz_result(score, weak_areas=None):
    roadmap = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[Topic(title="Intro", description="d", estimated_minutes=10)],
    )
    state = initial_state(goal="g", session_id="s")
    state["roadmap"] = roadmap
    state["current_topic_index"] = 0
    state["quiz_results"] = [
        QuizResult(
            topic="Intro", score=score, passed=score >= 0.5, weak_areas=weak_areas or []
        )
    ]
    state["messages"] = [AIMessage(content="Intro explanation.")]
    return state


def test_progress_coach_node_completes_topic_and_skips_study_buddy_on_pass(monkeypatch):
    state = _make_state_with_quiz_result(score=0.9)

    monkeypatch.setattr(
        progress_coach,
        "get_coaching_message",
        MagicMock(return_value=CoachingMessage(summary="Great job!", tip="Keep going")),
    )
    mock_study_buddy = MagicMock()
    monkeypatch.setattr(progress_coach, "try_study_buddy_assistance", mock_study_buddy)

    result = progress_coach_node(state)

    assert result["roadmap"].topics[0].status == "completed"
    assert result["current_topic_index"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Great job!"
    assert result["error"] is None
    mock_study_buddy.assert_not_called()


def test_progress_coach_node_requests_study_buddy_on_low_score(monkeypatch):
    state = _make_state_with_quiz_result(score=0.2, weak_areas=["recursion"])

    monkeypatch.setattr(
        progress_coach,
        "get_coaching_message",
        MagicMock(
            return_value=CoachingMessage(summary="Keep practicing.", tip="Review recursion")
        ),
    )
    mock_study_buddy = MagicMock(return_value="Here's a tip about recursion.")
    monkeypatch.setattr(progress_coach, "try_study_buddy_assistance", mock_study_buddy)

    result = progress_coach_node(state)

    assert result["roadmap"].topics[0].status == "needs_review"
    mock_study_buddy.assert_called_once_with(
        topic="Intro",
        explanation="Intro explanation.",
        weak_areas=["recursion"],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_progress_coach.py -v`
Expected: as in prior tasks — targets existing, unchanged behavior, should PASS if written correctly. A failure means the mocked `get_coaching_message`/`try_study_buddy_assistance` call shape doesn't match `progress_coach_node`'s actual code — check `src/learning_accelerator/agents/progress_coach.py` before changing the test.

- [ ] **Step 3: Confirm all 8 tests in the file pass**

Run: `uv run pytest tests/test_progress_coach.py -v`
Expected: PASS (8 passed — the original 6 plus these 2 new ones).

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests pass, count increased by exactly 2 over Task 3's count (7 more than the original baseline in total across all 4 tasks).

- [ ] **Step 5: Commit**

```bash
git add tests/test_progress_coach.py
git commit -m "$(cat <<'EOF'
Add unit tests for progress_coach_node

Covers the passing-score path (topic marked completed, Study Buddy
not consulted) and the failing-score-with-weak-areas path (Study
Buddy consulted with the right args, topic marked needs_review) —
progress_coach_node's own orchestration had zero fast-suite coverage
before this (its sub-functions were already tested individually).
EOF
)"
```
