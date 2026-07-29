# Chapter 9: The Complete System and What's Next — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the complete system into two real entry points — a
terminal CLI (`main.py`) and a Streamlit web UI (`streamlit_app.py`) — and
close out the course with deployment notes and final flashcards.

**Source material note:** same situation as chapters 7-8 — the article's
own page gave a paraphrased summary, not exact code. The companion
reference repo (`sandeepmb/freecodecamp-multi-agent-ai-system`) has the
actual `streamlit_app.py` and `main.py`, pulled via `gh api` and adapted
here.

**Architecture:** The key trick the reference uses — and this codebase
adopts unchanged — is compiling a *second* graph instance with
`interrupt_before=["quiz_generator"]`. Streamlit's rerun-based execution
model can't tolerate a blocking `input()` call inside a graph node (it
would hang the whole UI), so the UI-specific graph pauses *before*
`quiz_generator` runs, handles quiz question/answer/grading itself via
direct `generate_questions()`/`grade_answer()` calls, then injects the
result into the checkpoint with `graph.update_state(config, {...},
as_node="quiz_generator")` and resumes from `progress_coach` onward. This
requires zero changes to `quiz_generator_node` itself — only `main.py`'s
plain terminal flow uses the node's real `input()`-driven path.

**Deliberate adaptations from the reference repo:**
1. **`build_graph()` gains an `interrupt_before: list[str] | None = None`
   parameter**, threaded through to `builder.compile(...)`. Doesn't exist
   in this codebase yet (chapters 2-8 only ever built the module-level
   `graph` with no interrupt_before).
2. **A new `GradedAnswer` dataclass**, not a second `QuizQuestion`. The
   reference's `streamlit_app.py` imports a `QuizQuestion` from
   `graph.state` with fields `question/expected_answer/user_answer/
   correct/feedback/score` — but this codebase's `quiz_generator.py`
   *already* defines a `QuizQuestion` (Pydantic, `question/
   expected_answer/difficulty`, representing an *ungraded* generated
   question). Reusing the same name for a different, graded-answer shape
   would collide. `GradedAnswer` is the new name here.
3. **`QuizResult` gains a `questions: list[GradedAnswer] =
   field(default_factory=list)`** (chapter 4's version only had
   `topic/score/passed/weak_areas`) — additive, backward compatible,
   needed for the Streamlit Results screen's per-question breakdown.
   `quiz_generator_node`/`run_quiz()` (chapter 4, used by `main.py`'s
   plain terminal flow) are NOT changed to populate this field — only the
   Streamlit app needs per-question tracking, and it already builds
   `QuizResult` directly from its own accumulated session state rather
   than through `run_quiz()`.
4. **Pydantic objects assumed directly, no dict-or-object defensive
   branching.** The reference repo's code frequently does
   `topic.title if hasattr(topic, "title") else topic.get("title", "")`
   because its own checkpointer sometimes hands back plain dicts after a
   round-trip. Chapter 5 already proved this codebase's `StudyRoadmap`
   survives SQLite round-trips as a real Pydantic object (with only a
   deprecation warning, tracked separately) — so this code assumes
   `.attribute` access throughout, no dict fallback branching.
5. **`observability/langfuse_setup.py` gains a `flush_langfuse()`
   helper** — small, additive, needed by both new entry points to flush
   pending traces before process exit.
6. **Real verification of the Streamlit UI happens via actual browser
   automation** (this session's Browser tooling), not a subagent —
   walking through the real multi-screen flow against local Ollama,
   consistent with this repo's "start the dev server and use the feature
   in a browser" practice for UI work.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Out of scope (per spec): Appendix C production hardening checklist —
  this chapter's "deployment notes" stay a concise summary in flashcards/
  README, not a full production runbook
- Branch: `agent/chapter-9` (already created, off updated `main`)
- One PR per chapter; push, open, and merge it yourself once verified —
  full autonomy already authorized for chapters 5-9 (last chapter of the
  course)
- `build_graph`'s new `interrupt_before` parameter, `flush_langfuse()`,
  and the `GradedAnswer`/`QuizResult.questions` extension are all pure
  logic (compiling a graph or formatting a dataclass makes no network
  call) → full pytest coverage
- `main.py`'s `print_session_summary()` is pure formatting logic (given a
  result dict with real `StudyRoadmap`/`QuizResult` objects, no LLM call)
  → pytest-testable via `capsys`
- `streamlit_app.py` itself is not unit-tested (Streamlit's
  script-execution model isn't suited to pytest, and the reference repo
  itself has no test file for it either) — verified instead via a real
  browser walkthrough against local Ollama
- Local dev machine has Ollama with `gemma4:12b-mlx`, `gemma4:12b`,
  `qwen3.5:2b`; chapters 2-8 all found `gemma4:12b` works reliably
- `learning/chapterN.md` format: short Q/A flashcards (see chapters 1-8)

---

### Task 1: Foundation extensions (interrupt_before, flush_langfuse, GradedAnswer)

**Files:**
- Modify: `src/learning_accelerator/graph/workflow.py`
- Modify: `src/learning_accelerator/graph/state.py`
- Modify: `src/learning_accelerator/observability/langfuse_setup.py`
- Test: `tests/test_workflow.py` (new)
- Modify: `tests/test_state.py`
- Modify: `tests/test_langfuse_setup.py`

**Interfaces:**
- Produces: `build_graph(db_path=..., interrupt_before: list[str] | None
  = None)`, `GradedAnswer` dataclass, `QuizResult.questions` field,
  `flush_langfuse() -> None` — all consumed by Tasks 2-3

- [ ] **Step 1: Write the failing tests**

Create `tests/test_workflow.py`:

```python
from learning_accelerator.graph.workflow import build_graph


def test_build_graph_compiles_without_interrupt_before(tmp_path):
    graph = build_graph(db_path=str(tmp_path / "test.sqlite"))
    assert graph is not None


def test_build_graph_compiles_with_interrupt_before(tmp_path):
    graph = build_graph(
        db_path=str(tmp_path / "test.sqlite"),
        interrupt_before=["quiz_generator"],
    )
    assert graph is not None
```

Append to `tests/test_state.py`:

```python
from learning_accelerator.graph.state import GradedAnswer


def test_graded_answer_defaults():
    answer = GradedAnswer(
        question="What is a checkpoint?",
        expected_answer="A saved state snapshot.",
        user_answer="A save file.",
        correct=False,
        feedback="Close, but be more precise.",
        score=0.6,
    )
    assert answer.score == 0.6


def test_quiz_result_questions_defaults_to_empty_list():
    result = QuizResult(topic="LangGraph", score=0.8, passed=True)
    assert result.questions == []


def test_quiz_result_accepts_graded_answers():
    answer = GradedAnswer(
        question="Q", expected_answer="E", user_answer="U",
        correct=True, feedback="Good", score=1.0,
    )
    result = QuizResult(topic="LangGraph", score=1.0, passed=True, questions=[answer])
    assert result.questions[0].score == 1.0
```

(`QuizResult` is already imported at the top of `tests/test_state.py` —
verify and add `GradedAnswer` to the same import line if not already
present as a separate import.)

Append to `tests/test_langfuse_setup.py`:

```python
def test_flush_langfuse_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from learning_accelerator.observability.langfuse_setup import flush_langfuse

    flush_langfuse()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_workflow.py tests/test_state.py tests/test_langfuse_setup.py -v`
Expected: FAIL — `TypeError: build_graph() got an unexpected keyword argument 'interrupt_before'`, `ImportError: cannot import name 'GradedAnswer'`, `ImportError: cannot import name 'flush_langfuse'`

- [ ] **Step 3: Edit `src/learning_accelerator/graph/state.py`**

Add near `QuizResult`:

```python
@dataclass
class GradedAnswer:
    question: str
    expected_answer: str
    user_answer: str
    correct: bool
    feedback: str
    score: float
```

Update `QuizResult`:

```python
@dataclass
class QuizResult:
    topic: str
    score: float
    passed: bool
    weak_areas: list[str] = field(default_factory=list)
    questions: list[GradedAnswer] = field(default_factory=list)
```

- [ ] **Step 4: Edit `src/learning_accelerator/graph/workflow.py`**

Change:

```python
def build_graph(db_path: str = DEFAULT_DB_PATH):
```

to:

```python
def build_graph(
    db_path: str = DEFAULT_DB_PATH,
    interrupt_before: list[str] | None = None,
):
```

Change:

```python
    return builder.compile(checkpointer=checkpointer)
```

to:

```python
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
```

- [ ] **Step 5: Edit `src/learning_accelerator/observability/langfuse_setup.py`**

Add at the end of the file:

```python
def flush_langfuse() -> None:
    """Flush any pending Langfuse events before process exit.

    Langfuse batches/exports spans asynchronously; call this at the end
    of a script so all traces are actually sent before the process ends.
    No-op if Langfuse isn't configured.
    """
    if not langfuse_enabled():
        return

    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass  # best-effort flush, don't crash on exit
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_workflow.py tests/test_state.py tests/test_langfuse_setup.py -v`
Expected: all pass

- [ ] **Step 7: Run the full default suite**

Run: `uv run pytest -v`
Expected: all tests pass (this task only adds new fields/parameters with
defaults — no existing behavior changes)

- [ ] **Step 8: Commit**

```bash
git add src/learning_accelerator/graph/workflow.py src/learning_accelerator/graph/state.py src/learning_accelerator/observability/langfuse_setup.py tests/test_workflow.py tests/test_state.py tests/test_langfuse_setup.py
git commit -m "Add interrupt_before support, GradedAnswer, and flush_langfuse"
```

---

### Task 2: main.py CLI entry point + tests

**Files:**
- Create: `main.py` (repo root, alongside `pyproject.toml`)
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `graph` (module-level, chapter 2), `initial_state` (chapter
  2), `get_langfuse_config`/`flush_langfuse` (chapter 6/Task 1)
- Produces: `print_session_summary(result: dict) -> None`,
  `run_session(goal: str, session_id: str | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:

```python
from learning_accelerator.graph.state import QuizResult, StudyRoadmap, Topic


def test_print_session_summary_shows_average_and_topics(capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import print_session_summary

    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=2,
        topics=[
            Topic(title="Nodes and Edges", description="d", estimated_minutes=30),
            Topic(title="Checkpointing", description="d", estimated_minutes=30),
        ],
    )
    result = {
        "roadmap": roadmap,
        "quiz_results": [
            QuizResult(topic="Nodes and Edges", score=1.0, passed=True),
            QuizResult(topic="Checkpointing", score=0.4, passed=False, weak_areas=["thread_id"]),
        ],
        "weak_areas": ["thread_id"],
    }

    print_session_summary(result)

    captured = capsys.readouterr()
    assert "Learn LangGraph" in captured.out
    assert "Nodes and Edges" in captured.out
    assert "Checkpointing" in captured.out
    assert "70%" in captured.out  # average of 1.0 and 0.4
    assert "thread_id" in captured.out


def test_print_session_summary_handles_no_roadmap(capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import print_session_summary

    print_session_summary({"roadmap": None})

    captured = capsys.readouterr()
    assert captured.out == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `main.py`** (repo root)

```python
"""
main.py

Terminal entry point for the Learning Accelerator.

Usage:
  uv run python main.py "Learn LangGraph checkpointing from scratch"
  uv run python main.py --resume <session-id>
"""

from __future__ import annotations

import argparse
import uuid

from langgraph.types import Command

from learning_accelerator.graph.state import QuizResult, StudyRoadmap, initial_state
from learning_accelerator.graph.workflow import graph
from learning_accelerator.observability.langfuse_setup import flush_langfuse, get_langfuse_config


def print_session_summary(result: dict) -> None:
    """Print a summary of a completed session. No-op if there's no roadmap."""
    roadmap: StudyRoadmap | None = result.get("roadmap")
    if roadmap is None:
        return

    quiz_results: list[QuizResult] = result.get("quiz_results", [])
    if not quiz_results:
        return

    print(f"\n{'=' * 60}")
    print("Session Summary")
    print(f"{'=' * 60}")
    print(f"Goal: {roadmap.goal}")
    print(f"Topics covered: {len(quiz_results)}/{len(roadmap.topics)}")

    avg = sum(r.score for r in quiz_results) / len(quiz_results)
    print(f"Average score: {avg:.0%}\n")

    for r in quiz_results:
        status = "✓" if r.score >= 0.5 else "✗"
        weak = f", review: {', '.join(r.weak_areas)}" if r.weak_areas else ""
        print(f"  {status} {r.topic}: {r.score:.0%}{weak}")

    all_weak = result.get("weak_areas", [])
    if all_weak:
        print(f"\nTopics to revisit: {', '.join(all_weak)}")

    print(f"{'=' * 60}\n")


def run_session(goal: str, session_id: str | None = None) -> None:
    """Run a complete interactive study session with Langfuse tracing."""
    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    config = get_langfuse_config(session_id)

    print(f"\n{'=' * 60}")
    print("Learning Accelerator")
    print(f"Session ID: {session_id}")
    print("Resuming existing session..." if is_resume else f"Goal: {goal}")
    print(f"{'=' * 60}")

    state = None if is_resume else initial_state(goal, session_id)

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        if is_resume:
            print(f"\n[ERROR] Could not resume session '{session_id}': {e}")
            print("If the session ID is wrong or the checkpoint database has "
                  "been deleted, start a new session instead.")
            return
        raise

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        roadmap: StudyRoadmap | None = interrupt_payload.get("roadmap")

        if roadmap:
            print(f"\n{'=' * 60}")
            print("Proposed Study Plan")
            print(f"{'=' * 60}")
            print(f"Goal: {roadmap.goal}")
            print(f"Duration: {roadmap.total_weeks} weeks @ {roadmap.weekly_hours} hrs/week\n")
            for i, topic in enumerate(roadmap.topics, 1):
                prereqs = f" (needs: {', '.join(topic.prerequisites)})" if topic.prerequisites else ""
                print(f"  {i}. {topic.title} ({topic.estimated_minutes} min){prereqs}")
                print(f"     {topic.description}")

        print(f"\n{interrupt_payload.get('prompt', 'Continue?')}")
        user_input = input("> ").strip()

        result = graph.invoke(Command(resume=user_input), config=config)

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        return

    print_session_summary(result)
    flush_langfuse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Learning Accelerator: a four-agent study system that plans a "
            "curriculum, explains topics from your notes, quizzes you, and "
            "adapts based on results. All inference runs locally via Ollama "
            "by default."
        ),
        epilog=(
            "Examples:\n"
            '  uv run python main.py "Learn LangGraph checkpointing from scratch"\n'
            "  uv run python main.py --resume a3f1b2c4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="Learn the basics of LangGraph",
        help="What you want to learn (default: a LangGraph starter goal)",
    )
    parser.add_argument(
        "--resume", metavar="SESSION_ID", help="Resume an existing session by its ID"
    )
    args = parser.parse_args()

    if args.resume:
        run_session(goal="", session_id=args.resume)
    else:
        run_session(goal=args.goal)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_main.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full default suite**

Run: `uv run pytest -v`

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "Add main.py terminal entry point"
```

---

### Task 3: Streamlit web UI

**Files:**
- Create: `streamlit_app.py` (repo root)

**Interfaces:**
- Consumes: `build_graph` (Task 1), `initial_state`/`StudyRoadmap`/
  `QuizResult`/`GradedAnswer` (chapters 2/Task 1), `get_langfuse_config`/
  `flush_langfuse` (chapter 6/Task 1), `generate_questions`/`grade_answer`
  (chapter 4)

No pytest coverage for this file (script-execution model, see Global
Constraints) — verify by confirming it imports and its module-level
`ui_graph = build_graph(...)` line executes without error; full
interactive verification happens in Task 4.

- [ ] **Step 1: Add the streamlit dependency**

```bash
uv add streamlit
```

- [ ] **Step 2: Write `streamlit_app.py`** (repo root)

```python
"""
streamlit_app.py

Streamlit web interface for the Learning Accelerator.

Runs the same LangGraph graph as main.py — only the I/O mechanism
changes. Instead of terminal input/output, this uses Streamlit widgets
and session state.

Run:
    uv run streamlit run streamlit_app.py

Architecture:
    Five screens: GOAL_INPUT -> ROADMAP_APPROVAL -> EXPLAINING ->
    QUIZZING -> COMPLETE.

    A separate graph instance (ui_graph) is compiled with
    interrupt_before=["quiz_generator"] so the graph pauses before the
    quiz step and returns control to Streamlit. The UI handles quiz I/O
    directly (calling generate_questions and grade_answer), then injects
    the QuizResult into the checkpoint via ui_graph.update_state() and
    resumes execution from progress_coach onward.

    This means: zero changes to quiz_generator_node or run_quiz(); the
    terminal interface (main.py) is completely unaffected; the LangGraph
    graph code is identical, only I/O changes.
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage
from langgraph.types import Command

from learning_accelerator.agents.quiz_generator import generate_questions, grade_answer
from learning_accelerator.graph.state import GradedAnswer, QuizResult, initial_state
from learning_accelerator.graph.workflow import build_graph
from learning_accelerator.observability.langfuse_setup import flush_langfuse, get_langfuse_config

ui_graph = build_graph(
    db_path=".data/checkpoints_ui.sqlite",
    interrupt_before=["quiz_generator"],
)

st.set_page_config(page_title="Learning Accelerator", page_icon="🎓", layout="centered")


def init_state() -> None:
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "roadmap": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "quiz_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "coaching_message": "",
        "error": None,
        "goal": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def go_to(screen: str) -> None:
    st.session_state.screen = screen


def extract_explanation(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""


def extract_coaching(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def new_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def start_session(goal: str) -> None:
    """Runs: curriculum_planner -> human_approval (interrupt)."""
    session_id = str(uuid.uuid4())[:8]
    config = get_langfuse_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config
    st.session_state.goal = goal

    state = initial_state(goal, session_id)

    with st.spinner("Building your study roadmap..."):
        result = ui_graph.invoke(state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
    elif result.get("error"):
        st.session_state.error = result["error"]
    else:
        st.session_state.error = "Unexpected: no interrupt after planner."


def approve_roadmap(approved: bool) -> None:
    """If approved: human_approval -> explainer, then pauses before quiz_generator.
    If rejected: human_approval -> curriculum_planner -> interrupt again."""
    decision = "yes" if approved else "no"

    with st.spinner("Starting your study session..." if approved else "Generating a new plan..."):
        result = ui_graph.invoke(Command(resume=decision), config=st.session_state.graph_config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
        return

    messages = result.get("messages", [])
    st.session_state.explanation = extract_explanation(messages)

    roadmap = result.get("roadmap") or st.session_state.roadmap
    st.session_state.roadmap = roadmap
    idx = result.get("current_topic_index", 0)
    st.session_state.current_topic_index = idx

    topic = roadmap.topics[idx]
    st.session_state.topic_title = topic.title
    st.session_state.topic_description = topic.description

    with st.spinner("Generating quiz questions..."):
        questions = generate_questions(topic.title, st.session_state.explanation, n=3)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []

    go_to("EXPLAINING")


def advance_after_quiz(quiz_result: QuizResult) -> None:
    """Inject the QuizResult as if quiz_generator ran, then resume from
    progress_coach onward."""
    config = st.session_state.graph_config
    existing = st.session_state.quiz_results
    all_weak = list(set(st.session_state.weak_areas + quiz_result.weak_areas))

    ui_graph.update_state(
        config,
        {
            "quiz_results": existing + [quiz_result],
            "weak_areas": all_weak,
            "roadmap": st.session_state.roadmap,
            "current_topic_index": st.session_state.current_topic_index,
            "error": None,
        },
        as_node="quiz_generator",
    )

    with st.spinner("Getting coaching feedback..."):
        result = ui_graph.invoke(None, config=config)

    messages = result.get("messages", [])
    st.session_state.coaching_message = extract_coaching(messages)
    st.session_state.quiz_results = result.get("quiz_results", existing + [quiz_result])
    st.session_state.weak_areas = result.get("weak_areas", all_weak)
    new_idx = result.get("current_topic_index", st.session_state.current_topic_index + 1)
    st.session_state.current_topic_index = new_idx

    roadmap = result.get("roadmap", st.session_state.roadmap)
    st.session_state.roadmap = roadmap

    if roadmap is None or new_idx >= len(roadmap.topics):
        flush_langfuse()
        go_to("COMPLETE")
        return

    st.session_state.explanation = extract_explanation(messages)
    topic = roadmap.topics[new_idx]
    st.session_state.topic_title = topic.title
    st.session_state.topic_description = topic.description

    with st.spinner("Generating quiz questions..."):
        questions = generate_questions(topic.title, st.session_state.explanation, n=3)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []

    go_to("EXPLAINING")


def screen_goal_input() -> None:
    st.title("🎓 Learning Accelerator")
    st.markdown(
        "Enter a learning goal and the system will build a personalised "
        "study plan, explain each topic using your notes, and quiz you "
        "as you go, all running locally with Ollama."
    )

    with st.form("goal_form"):
        goal = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. Learn LangGraph checkpointing from scratch",
        )
        submitted = st.form_submit_button("Build Study Plan →", type="primary")

    if submitted:
        if not goal.strip():
            st.error("Please enter a learning goal.")
        else:
            start_session(goal.strip())
            st.rerun()

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Try again"):
            st.session_state.error = None
            st.rerun()


def screen_roadmap_approval() -> None:
    st.title("📋 Your Study Plan")
    roadmap = st.session_state.roadmap

    if roadmap is None:
        st.error("No roadmap found.")
        if st.button("Start over"):
            new_session()
            st.rerun()
        return

    st.markdown(f"**Goal:** {roadmap.goal}")
    st.markdown(f"**Duration:** {roadmap.total_weeks} weeks @ {roadmap.weekly_hours} hrs/week")
    st.markdown("---")

    for i, topic in enumerate(roadmap.topics, 1):
        prereq_text = f" *(needs: {', '.join(topic.prerequisites)})*" if topic.prerequisites else ""
        st.markdown(f"**{i}. {topic.title}**, {topic.estimated_minutes} min{prereq_text}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{topic.description}")

    st.markdown("---")
    st.markdown("Does this study plan look good?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, start studying", type="primary", use_container_width=True):
            approve_roadmap(True)
            st.rerun()
    with col2:
        if st.button("🔄 No, generate a different plan", use_container_width=True):
            approve_roadmap(False)
            st.rerun()


def screen_explaining() -> None:
    roadmap = st.session_state.roadmap
    total = len(roadmap.topics) if roadmap else 1
    idx = st.session_state.current_topic_index

    st.progress(idx / total, text=f"Topic {idx + 1} of {total}")
    st.title(f"📖 {st.session_state.topic_title}")
    st.caption(st.session_state.topic_description)
    st.markdown("---")

    if st.session_state.coaching_message:
        st.info(f"💬 **Coach:** {st.session_state.coaching_message}")
        st.markdown("---")

    if st.session_state.explanation:
        st.markdown("### Explanation")
        st.markdown(st.session_state.explanation)
    else:
        st.warning("No explanation available, starting quiz with topic context.")

    st.markdown("---")
    st.markdown(f"**Ready to test your knowledge of *{st.session_state.topic_title}*?**")

    if st.button("Start Quiz →", type="primary"):
        st.session_state.coaching_message = ""
        go_to("QUIZZING")
        st.rerun()


def screen_quizzing() -> None:
    questions = st.session_state.quiz_questions
    q_idx = st.session_state.current_question_idx
    total_q = len(questions)
    roadmap = st.session_state.roadmap
    total_topics = len(roadmap.topics) if roadmap else 1
    topic_idx = st.session_state.current_topic_index

    st.progress(topic_idx / total_topics, text=f"Topic {topic_idx + 1} of {total_topics}")
    if total_q > 0:
        st.progress(q_idx / total_q, text=f"Question {q_idx + 1} of {total_q}")

    st.title(f"🧠 Quiz: {st.session_state.topic_title}")
    st.markdown("---")

    for i, graded in enumerate(st.session_state.graded_answers):
        status = "✅" if graded.correct else "❌"
        with st.expander(f"{status} Q{i + 1}: {graded.question[:80]}...", expanded=False):
            st.markdown(f"**Your answer:** {graded.user_answer}")
            st.markdown(f"**Score:** {graded.score:.0%}")
            st.markdown(f"**Feedback:** {graded.feedback}")

    if q_idx < total_q:
        q = questions[q_idx]

        st.markdown(f"**Question {q_idx + 1} [{q.difficulty}]:**")
        st.markdown(q.question)

        with st.form(f"answer_form_{q_idx}"):
            answer = st.text_area(
                "Your answer:", placeholder="Type your answer here...",
                height=120, key=f"answer_input_{q_idx}",
            )
            submitted = st.form_submit_button("Submit Answer →", type="primary")

        if submitted:
            user_answer = answer.strip() or "(no answer provided)"

            with st.spinner("Grading your answer..."):
                grade = grade_answer(q.question, q.expected_answer, user_answer)

            graded_answer = GradedAnswer(
                question=q.question,
                expected_answer=q.expected_answer,
                user_answer=user_answer,
                correct=grade.correct,
                feedback=grade.feedback,
                score=grade.score,
            )
            st.session_state.graded_answers.append(graded_answer)
            if grade.missing_concept:
                st.session_state.current_quiz_missing_concepts.append(grade.missing_concept)
            st.session_state.current_question_idx = q_idx + 1
            st.rerun()

    else:
        st.markdown("---")
        graded = st.session_state.graded_answers
        avg_score = sum(a.score for a in graded) / len(graded) if graded else 0.0
        weak_areas = list(dict.fromkeys(st.session_state.current_quiz_missing_concepts))

        st.success("✅ Quiz complete!")
        st.metric("Your score", f"{avg_score:.0%}")

        quiz_result = QuizResult(
            topic=st.session_state.topic_title,
            score=avg_score,
            passed=avg_score >= 0.5,
            weak_areas=weak_areas,
            questions=graded,
        )

        if st.button("Continue →", type="primary"):
            advance_after_quiz(quiz_result)
            st.rerun()


def screen_complete() -> None:
    st.title("🎉 Session Complete!")
    st.markdown("---")

    roadmap = st.session_state.roadmap
    quiz_results = st.session_state.quiz_results

    if roadmap:
        st.markdown(f"**Goal:** {roadmap.goal}")

    if quiz_results:
        avg = sum(r.score for r in quiz_results) / len(quiz_results)
        st.metric("Overall Average", f"{avg:.0%}")
        st.markdown("---")
        st.markdown("### Results by Topic")
        for r in quiz_results:
            status = "✅" if r.score >= 0.5 else "❌"
            weak = f", review: {', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
            st.markdown(f"{status} **{r.topic}**: {r.score:.0%}{weak}")

    if st.session_state.weak_areas:
        st.markdown("---")
        st.markdown("### Topics to Revisit")
        for w in st.session_state.weak_areas[:5]:
            st.markdown(f"- {w}")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Start a New Session", type="primary"):
        new_session()
        st.rerun()


def display_error() -> None:
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")
        if st.button("← Start over"):
            new_session()
            st.rerun()


screen = st.session_state.screen

if screen == "GOAL_INPUT":
    screen_goal_input()
elif screen == "ROADMAP_APPROVAL":
    display_error()
    screen_roadmap_approval()
elif screen == "EXPLAINING":
    display_error()
    screen_explaining()
elif screen == "QUIZZING":
    display_error()
    screen_quizzing()
elif screen == "COMPLETE":
    screen_complete()
else:
    st.error(f"Unknown screen: {screen}")
    if st.button("Reset"):
        new_session()
        st.rerun()
```

- [ ] **Step 3: Verify it imports and the module-level graph builds**

Run: `rm -f .data/checkpoints_ui.sqlite && uv run python -c "import streamlit_app" 2>&1 | tail -20`

This will likely print Streamlit's "missing ScriptRunContext" warnings
(harmless — that's expected when importing outside `streamlit run`) but
must NOT raise an actual exception. If it does raise, fix the import/
syntax issue before proceeding.

- [ ] **Step 4: Run the full pytest suite**

Run: `uv run pytest -v`
Expected: all tests pass (adding `streamlit_app.py` and the `streamlit`
dependency shouldn't affect any existing test)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock streamlit_app.py
git commit -m "Add Streamlit web UI"
```

---

### Task 4: Real browser-driven verification (controller does this directly, not a subagent)

Start the Streamlit app for real (`uv run streamlit run streamlit_app.py`)
via this session's dev-server preview tooling, then drive it through the
Browser tool for a genuine end-to-end walkthrough against local Ollama:

1. Submit a learning goal on the Goal Input screen
2. Approve the generated roadmap
3. Read the Explainer's output on the Explaining screen, start the quiz
4. Answer at least one quiz question for real, see it graded
5. Continue through to either the next topic or the Complete screen
6. Confirm the Complete screen shows a real average score and topic
   breakdown

Record the actual observed behavior (including any bugs found and fixed)
for the chapter 9 flashcards in Task 6. If something breaks, fix it
directly (this is real UI verification, not a scripted demo) — likely
candidates given chapter 8's A2A findings: `ui_graph.update_state(...,
as_node=...)`'s exact signature/behavior should be double-checked against
the installed `langgraph` version if it errors.

---

### Task 5: README update — tying the complete system together

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Running the System" section**

Add a section after "Getting Started" (or wherever fits best) covering:
- `uv run python main.py "<your learning goal>"` — terminal interface
- `uv run python main.py --resume <session-id>` — resume a session
- `uv run streamlit run streamlit_app.py` — web interface
- A one-line pointer to `learning/chapter1.md` through `chapter9.md` for
  the flashcard-style course notes, and to `docs/architecture.md` for the
  multi-agent design rationale

Keep it brief — a few lines, not a full user guide.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document how to run the complete system"
```

---

### Task 6: Chapter 9 learning flashcards

**Files:**
- Create: `learning/chapter9.md`

- [ ] **Step 1: Write `learning/chapter9.md`**

Cover at minimum: why the Streamlit UI needs a *second* graph instance
with `interrupt_before=["quiz_generator"]` rather than reusing the
module-level `graph`, how `update_state(..., as_node="quiz_generator")`
lets the UI "fake" a node having run, why `quiz_generator_node` itself
needed zero changes to support this, the real results from Task 4's
browser walkthrough (fill in with what was actually observed — don't
leave placeholder text), a concise summary of this course's production-
relevant patterns (observability via Langfuse, quality evaluation via
DeepEval, cross-framework coordination via A2A — the actual chapters 6-8
built these, this isn't hypothetical), and why deployment hardening
(Appendix C equivalent) stays out of this chapter's scope per the
project's own spec.

- [ ] **Step 2: Commit**

```bash
git add learning/chapter9.md
git commit -m "Add chapter 9 learning flashcards"
```

---

### Task 7: Push, open, and merge the chapter 9 PR (final chapter)

**Files:** none (branch/PR operation only)

Full autonomy is authorized for chapters 5-9 — no confirmation needed
before push/PR/merge. Before this task, dispatch a final whole-branch
review (per subagent-driven-development) covering all of Tasks 1-6
together, and address any Critical/Important findings before proceeding.

- [ ] **Step 1: Confirm the default fast suite passes**

Run: `uv run pytest -v`

- [ ] **Step 2: Push the branch**

```bash
git push -u origin agent/chapter-9
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "Chapter 9: The Complete System and What's Next" --body "..."
```

- [ ] **Step 4: Merge**

```bash
gh pr merge --merge
```

- [ ] **Step 5: Delete the remote branch and sync the local worktree**

```bash
git push origin --delete agent/chapter-9
git checkout claude/agentic-ai-langgraph-course-ebf06b
git fetch origin
git merge --ff-only origin/main
git branch -d agent/chapter-9
```

This is the final chapter — after this merges, the 9-chapter course is complete.
