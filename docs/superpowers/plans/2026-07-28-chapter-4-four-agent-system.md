# Chapter 4: Building the Four-Agent System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Quiz Generator and Progress Coach agents, wire conditional
routing that loops through every roadmap topic, and run the full four-agent
system end to end.

**Architecture:** `agents/quiz_generator.py` adds two LLM calls at different
temperatures (question generation `0.4`, grading `0.1`) plus `run_quiz()`
orchestration and `quiz_generator_node`. `agents/progress_coach.py` adds a
coaching-message LLM call (`0.4`), topic status transition, memory
persistence, and the loop-or-end routing function. `graph/state.py` gains
two small pure-logic helpers (`get_current_topic`, `session_is_complete`)
shared across nodes, plus a `PASS_THRESHOLD` constant and an extended
`QuizResult` (adds `weak_areas`, matching the article's actual usage —
chapter 2's version was missing it). `graph/workflow.py` extends to
`curriculum_planner → explainer → quiz_generator → progress_coach →
(conditional)→ explainer | END`.

**Deliberate deviations from the article, both already agreed with the
user:**
1. **No `human_approval` node yet.** The article's chapter 4 code already
   includes it, but our spec assigns it to chapter 5 — this chapter keeps
   the direct `curriculum_planner → explainer` edge from chapter 3.
2. **Injectable `answer_source` in `run_quiz`.** The article calls real
   `input()` directly. We can't drive real stdin through this session's
   tooling to verify the full loop end-to-end, so `run_quiz` takes an
   optional `answer_source: Callable[[str], str] | None = None` (defaults
   to real `input` when `None` — identical behavior for genuine
   interactive use) so the chapter 4 demo script can supply canned
   answers instead.

**Tech Stack:** LangChain `with_structured_output` (Pydantic schemas, same
pattern as Curriculum Planner), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-4` (already created, off updated `main`)
- One PR per chapter, merged before chapter 5 starts
- `PASS_THRESHOLD = 0.5` lives once in `graph/state.py`, imported by both
  `quiz_generator.py` (to set `QuizResult.passed`) and
  `progress_coach.py` (to set topic status) — don't redefine it twice
- Testing approach (from spec): pytest for pure logic only.
  `get_current_topic`, `session_is_complete`, `next_topic_status`, and
  `route_after_coach` are pure (no LLM calls) → pytest. `generate_questions`,
  `grade_answer`, `get_coaching_message` are LLM-calling → manual-run only
- Local dev machine has Ollama with `gemma4:12b-mlx`, `gemma4:12b`,
  `qwen3.5:2b`; chapters 2-3 found `gemma4:12b` works for both structured
  output and tool-calling — use it as the default verification model here
  too, but confirm rather than assume
- `learning/chapterN.md` format: short Q/A flashcards (see chapters 1-3)

---

### Task 1: Extend shared state (`graph/state.py`)

**Files:**
- Modify: `src/learning_accelerator/graph/state.py`
- Modify: `tests/test_state.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `PASS_THRESHOLD: float`, `QuizResult.weak_areas: list[str]`
  (new field, defaults to `[]`), `get_current_topic(state) -> Topic`,
  `session_is_complete(state) -> bool` — used by `explainer.py` (Task 2),
  `quiz_generator.py`/`progress_coach.py` (Tasks 3-4), `workflow.py`
  (Task 5)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
from learning_accelerator.graph.state import get_current_topic, session_is_complete


def test_get_current_topic_returns_indexed_topic():
    roadmap = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[
            Topic(title="A", description="d", estimated_minutes=10),
            Topic(title="B", description="d", estimated_minutes=10),
        ],
    )
    state = initial_state(goal="g", session_id="s")
    state["roadmap"] = roadmap
    state["current_topic_index"] = 1

    assert get_current_topic(state).title == "B"


def test_session_is_complete_true_when_no_roadmap():
    state = initial_state(goal="g", session_id="s")
    assert session_is_complete(state) is True


def test_session_is_complete_false_when_topics_remain():
    roadmap = StudyRoadmap(
        goal="g", total_weeks=1,
        topics=[Topic(title="A", description="d", estimated_minutes=10)],
    )
    state = initial_state(goal="g", session_id="s")
    state["roadmap"] = roadmap

    assert session_is_complete(state) is False


def test_session_is_complete_true_when_index_exceeds_topics():
    roadmap = StudyRoadmap(
        goal="g", total_weeks=1,
        topics=[Topic(title="A", description="d", estimated_minutes=10)],
    )
    state = initial_state(goal="g", session_id="s")
    state["roadmap"] = roadmap
    state["current_topic_index"] = 1

    assert session_is_complete(state) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'get_current_topic'`

- [ ] **Step 3: Edit `src/learning_accelerator/graph/state.py`**

Add near the top (after imports):

```python
PASS_THRESHOLD = 0.5
```

Replace the `QuizResult` dataclass:

```python
@dataclass
class QuizResult:
    topic: str
    score: float
    passed: bool
    weak_areas: list[str] = field(default_factory=list)
```

(This requires changing the `dataclass` import line to
`from dataclasses import dataclass, field`.)

Add at the end of the file:

```python
def get_current_topic(state: AgentState) -> Topic:
    """Return the topic the graph is currently working through.

    Only valid while session_is_complete(state) is False.
    """
    return state["roadmap"].topics[state["current_topic_index"]]


def session_is_complete(state: AgentState) -> bool:
    roadmap = state.get("roadmap")
    if roadmap is None:
        return True
    return state.get("current_topic_index", 0) >= len(roadmap.topics)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/graph/state.py tests/test_state.py
git commit -m "Extend shared state: QuizResult.weak_areas, get_current_topic, session_is_complete"
```

---

### Task 2: Refactor Explainer to use `get_current_topic`

**Files:**
- Modify: `src/learning_accelerator/agents/explainer.py`

**Interfaces:**
- Consumes: `get_current_topic` (Task 1)
- Produces: no change to `explainer_node`'s external behavior

- [ ] **Step 1: Edit `src/learning_accelerator/agents/explainer.py`**

Change the import line:

```python
from learning_accelerator.graph.state import AgentState
```

to:

```python
from learning_accelerator.graph.state import AgentState, get_current_topic
```

Replace:

```python
    topic = state["roadmap"].topics[state["current_topic_index"]]
```

with:

```python
    topic = get_current_topic(state)
```

- [ ] **Step 2: Run the full test suite to confirm no regression**

Run: `uv run pytest -v`
Expected: all 26 tests pass (this is a pure refactor, no new tests)

- [ ] **Step 3: Commit**

```bash
git add src/learning_accelerator/agents/explainer.py
git commit -m "Use shared get_current_topic helper in Explainer"
```

---

### Task 3: Quiz Generator agent

**Files:**
- Create: `src/learning_accelerator/agents/quiz_generator.py`

**Interfaces:**
- Consumes: `get_chat_model` (chapter 2), `AgentState`/`QuizResult`/
  `get_current_topic`/`PASS_THRESHOLD` (Task 1)
- Produces: `generate_questions`, `grade_answer`, `run_quiz`,
  `quiz_generator_node(state) -> dict`, wired into the graph in Task 5

No unit test — LLM-calling, excluded from pytest per the spec's testing
approach. Verified for real in Task 6.

- [ ] **Step 1: Write `src/learning_accelerator/agents/quiz_generator.py`**

```python
from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import (
    PASS_THRESHOLD,
    AgentState,
    QuizResult,
    get_current_topic,
)

GENERATION_PROMPT = """You are a quiz designer for a student learning \
programming. Given a topic and explanation, generate quiz questions that \
test genuine understanding, not rote recall.

Rules:
- Each question must end with "?".
- "expected_answer" is a model answer in 1-3 sentences.
- "difficulty" is one of "easy", "medium", "hard".
"""

GRADING_PROMPT = """You are a fair teacher grading a student's answer. \
Compare the student's answer to the model answer and grade honestly — \
partial credit is fine.

Rules:
- "score" is a float between 0.0 and 1.0.
- "correct" is true only if the score is 1.0.
- "feedback" is one specific sentence.
- "missing_concept" is the key concept the student missed, or "" if the
  answer is correct.
"""


class QuizQuestion(BaseModel):
    question: str
    expected_answer: str
    difficulty: str = "medium"


class QuestionSet(BaseModel):
    questions: list[QuizQuestion]


class GradeResult(BaseModel):
    correct: bool
    score: float
    feedback: str
    missing_concept: str = ""


AnswerSource = Callable[[str], str]


def _default_answer_source(question: str) -> str:
    return input(f"{question}\nYour answer: ").strip()


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[QuizQuestion]:
    llm = get_chat_model(temperature=0.4).with_structured_output(QuestionSet)
    result = llm.invoke(
        [
            {"role": "system", "content": GENERATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\nExplanation: {explanation}\n"
                    f"Generate exactly {n} questions."
                ),
            },
        ]
    )
    return result.questions


def grade_answer(question: str, expected: str, student_answer: str) -> GradeResult:
    llm = get_chat_model(temperature=0.1).with_structured_output(GradeResult)
    return llm.invoke(
        [
            {"role": "system", "content": GRADING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nModel answer: {expected}\n"
                    f"Student's answer: {student_answer}"
                ),
            },
        ]
    )


def run_quiz(
    topic: str,
    explanation: str,
    n: int = 3,
    answer_source: AnswerSource | None = None,
) -> QuizResult:
    ask = answer_source or _default_answer_source
    questions = generate_questions(topic, explanation, n=n)

    total_score = 0.0
    weak_areas: list[str] = []

    for q in questions:
        student_answer = ask(q.question)
        grade = grade_answer(q.question, q.expected_answer, student_answer)
        total_score += grade.score
        if grade.missing_concept:
            weak_areas.append(grade.missing_concept)

    avg_score = total_score / len(questions) if questions else 0.0

    return QuizResult(
        topic=topic,
        score=avg_score,
        passed=avg_score >= PASS_THRESHOLD,
        weak_areas=weak_areas,
    )


def quiz_generator_node(state: AgentState) -> dict:
    topic = get_current_topic(state)

    explanation = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            explanation = msg.content
            break

    quiz_result = run_quiz(topic.title, explanation)

    all_weak_areas = list(set(state.get("weak_areas", []) + quiz_result.weak_areas))

    return {
        "quiz_results": state.get("quiz_results", []) + [quiz_result],
        "weak_areas": all_weak_areas,
        "error": None,
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/learning_accelerator/agents/quiz_generator.py
git commit -m "Add Quiz Generator agent"
```

---

### Task 4: Progress Coach agent + routing

**Files:**
- Create: `src/learning_accelerator/agents/progress_coach.py`
- Test: `tests/test_progress_coach.py`

**Interfaces:**
- Consumes: `get_chat_model` (chapter 2), `memory_set` (chapter 3),
  `AgentState`/`PASS_THRESHOLD`/`session_is_complete` (Task 1)
- Produces: `get_coaching_message`, `next_topic_status(score) -> str`,
  `progress_coach_node(state) -> dict`, `route_after_coach(state) -> str`
  — the latter two wired into the graph in Task 5

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress_coach.py`:

```python
from learning_accelerator.agents.progress_coach import (
    next_topic_status,
    route_after_coach,
)
from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state


def test_next_topic_status_completed_at_threshold():
    assert next_topic_status(0.5) == "completed"


def test_next_topic_status_completed_above_threshold():
    assert next_topic_status(0.9) == "completed"


def test_next_topic_status_needs_review_below_threshold():
    assert next_topic_status(0.49) == "needs_review"


def test_route_after_coach_continues_when_topics_remain():
    roadmap = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[
            Topic(title="A", description="d", estimated_minutes=10),
            Topic(title="B", description="d", estimated_minutes=10),
        ],
    )
    state = initial_state(goal="g", session_id="s")
    state["roadmap"] = roadmap
    state["current_topic_index"] = 1

    assert route_after_coach(state) == "explainer"


def test_route_after_coach_ends_when_topics_exhausted():
    roadmap = StudyRoadmap(
        goal="g", total_weeks=1,
        topics=[Topic(title="A", description="d", estimated_minutes=10)],
    )
    state = initial_state(goal="g", session_id="s")
    state["roadmap"] = roadmap
    state["current_topic_index"] = 1

    assert route_after_coach(state) == "end"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_progress_coach.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.agents.progress_coach'`

- [ ] **Step 3: Write `src/learning_accelerator/agents/progress_coach.py`**

```python
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import AgentState, PASS_THRESHOLD, session_is_complete
from learning_accelerator.mcp_servers.memory_server import memory_set

COACHING_PROMPT = """You are a warm, encouraging study coach. Given a topic, \
a score, and any weak areas, write a short encouraging summary and one \
concrete tip for what to review next.
"""


class CoachingMessage(BaseModel):
    summary: str
    tip: str


def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> CoachingMessage:
    llm = get_chat_model(temperature=0.4).with_structured_output(CoachingMessage)
    context = {
        "topic": topic,
        "score_percent": f"{score:.0%}",
        "weak_areas": weak_areas if weak_areas else ["none identified"],
    }
    return llm.invoke(
        [
            SystemMessage(content=COACHING_PROMPT),
            HumanMessage(content=json.dumps(context)),
        ]
    )


def next_topic_status(score: float) -> str:
    return "completed" if score >= PASS_THRESHOLD else "needs_review"


def progress_coach_node(state: AgentState) -> dict:
    quiz_results = state.get("quiz_results", [])
    latest = quiz_results[-1]
    roadmap = state["roadmap"]
    idx = state.get("current_topic_index", 0)
    session_id = state["session_id"]

    coaching = get_coaching_message(latest.topic, latest.score, latest.weak_areas)

    if idx < len(roadmap.topics):
        roadmap.topics[idx].status = next_topic_status(latest.score)

    memory_set(
        session_id,
        f"progress_topic_{idx}",
        json.dumps(
            {
                "topic": latest.topic,
                "score": latest.score,
                "weak_areas": latest.weak_areas,
            }
        ),
    )

    return {
        "roadmap": roadmap,
        "current_topic_index": idx + 1,
        "messages": [AIMessage(content=coaching.summary)],
        "error": None,
    }


def route_after_coach(state: AgentState) -> str:
    return "end" if session_is_complete(state) else "explainer"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_progress_coach.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/agents/progress_coach.py tests/test_progress_coach.py
git commit -m "Add Progress Coach agent and loop routing"
```

---

### Task 5: Wire Quiz Generator and Progress Coach into the graph

**Files:**
- Modify: `src/learning_accelerator/graph/workflow.py`

**Interfaces:**
- Consumes: `quiz_generator_node` (Task 3), `progress_coach_node`/
  `route_after_coach` (Task 4)
- Produces: graph now runs `START → curriculum_planner → explainer →
  quiz_generator → progress_coach →(conditional)→ explainer | END`

- [ ] **Step 1: Edit `src/learning_accelerator/graph/workflow.py`**

Add imports:

```python
from learning_accelerator.agents.progress_coach import (
    progress_coach_node,
    route_after_coach,
)
from learning_accelerator.agents.quiz_generator import quiz_generator_node
```

Replace:

```python
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("explainer", explainer_node)
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "explainer")
    builder.add_edge("explainer", END)
```

with:

```python
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("explainer", explainer_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("progress_coach", progress_coach_node)

    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "explainer")
    builder.add_edge("explainer", "quiz_generator")
    builder.add_edge("quiz_generator", "progress_coach")
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {"explainer": "explainer", "end": END},
    )
```

- [ ] **Step 2: Verify the graph still compiles**

Run: `rm -f .data/checkpoints.sqlite && uv run python -c "from learning_accelerator.graph.workflow import graph; print(graph)"`
Expected: prints a `CompiledStateGraph` object, no errors

- [ ] **Step 3: Run the full pytest suite**

Run: `uv run pytest -v`
Expected: all 31 tests pass (9 state + 5 config + 7 filesystem_server + 5
memory_server + 5 progress_coach — this task only changes graph wiring)

- [ ] **Step 4: Commit**

```bash
git add src/learning_accelerator/graph/workflow.py
git commit -m "Wire Quiz Generator and Progress Coach into graph with loop routing"
```

---

### Task 6: Manual demo script + real end-to-end verification run

**Files:**
- Create: `scripts/demo_chapter4.py`

**Interfaces:**
- Consumes: `graph`, `initial_state` (chapter 2/Task 5),
  `quiz_generator._default_answer_source` (Task 3, monkeypatched here)

- [ ] **Step 1: Write `scripts/demo_chapter4.py`**

```python
"""Manual run: invoke the full chapter 4 graph (all four agents), looping
through every roadmap topic to completion.

This demo monkeypatches quiz_generator._default_answer_source with a
canned response so the run completes without a real terminal attached —
there's no way to feed this session's tooling real interactive stdin. For
a genuine interactive quiz, call run_quiz() directly from a real terminal
instead (its default answer_source is real input()).

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly.
"""

from __future__ import annotations

import uuid

from learning_accelerator.agents import quiz_generator
from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import graph


def _canned_answer_source(question: str) -> str:
    print(f"[canned] {question}")
    return "I'm not fully sure, but I'll give it my best guess."


def main() -> None:
    quiz_generator._default_answer_source = _canned_answer_source

    session_id = str(uuid.uuid4())
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke(state, config=config)

    print(f"Session: {session_id}")
    print(f"Final topic index: {result['current_topic_index']}")
    for topic in result["roadmap"].topics:
        print(f"- {topic.title}: {topic.status}")
    print(f"Quiz results: {len(result['quiz_results'])}")
    for qr in result["quiz_results"]:
        print(f"  {qr.topic}: score={qr.score:.2f} passed={qr.passed}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real against local Ollama**

This runs every topic in the roadmap through explain → quiz → coach, so
expect several minutes of local inference (run in the background if the
session's tooling supports it, rather than blocking).

```bash
rm -f .data/checkpoints.sqlite
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b uv run python scripts/demo_chapter4.py
```

Expected: prints a session ID, every topic with a final status of
`completed` or `needs_review` (not `pending`), and one quiz result per
topic. If a different Ollama model is needed (structured output or
tool-calling failures), note which one in the chapter 4 flashcards, same
as chapters 2-3.

- [ ] **Step 3: Run the full pytest suite once more**

Run: `uv run pytest -v`
Expected: all tests still pass (this step only adds a script).

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_chapter4.py
git commit -m "Add chapter 4 manual demo script"
```

---

### Task 7: Chapter 4 learning flashcards

**Files:**
- Create: `learning/chapter4.md`

- [ ] **Step 1: Write `learning/chapter4.md`**

```markdown
## Chapter 4: Building the Four-Agent System

**Q: Why does the Quiz Generator use two different LLM calls instead of one?**
A: Generating questions wants creative variety (`temperature=0.4`);
grading answers wants consistent, analytical scoring (`temperature=0.1`).
One shared temperature would compromise both.

**Q: What determines whether a topic is marked "completed" or
"needs_review"?**
A: `next_topic_status(score)` — `"completed"` if the quiz average score
is `>= PASS_THRESHOLD` (0.5), otherwise `"needs_review"`.

**Q: When does `current_topic_index` advance — before or after the topic
status update?**
A: After. `progress_coach_node` sets `roadmap.topics[idx].status` first,
then returns `current_topic_index: idx + 1`.

**Q: How does the graph know when to stop looping through topics?**
A: `route_after_coach` calls `session_is_complete(state)`, which checks
`current_topic_index >= len(roadmap.topics)`. True routes to `END`;
false routes back to `"explainer"` for the next topic.

**Q: Why does `run_quiz` take an injectable `answer_source` parameter
instead of always calling `input()` directly like the article?**
A: The article's version blocks on real interactive stdin, which can't be
driven from this session's tooling. `answer_source` defaults to real
`input` for genuine interactive use, but the chapter 4 demo swaps in a
canned answer function so the full graph can run end-to-end without a
live terminal.

**Q: Why doesn't chapter 4's graph include `human_approval` yet, even
though the source article's chapter 4 code does?**
A: Our own spec deliberately assigns `human_approval` + `interrupt()` to
chapter 5. This chapter keeps the direct `curriculum_planner → explainer`
edge from chapter 3; chapter 5 inserts `human_approval` between them.

**Q: What does chapter 4's graph look like now?**
A: `START → curriculum_planner → explainer → quiz_generator →
progress_coach →(conditional)→ explainer | END`.

**Q: Which local Ollama model was used to verify the full end-to-end loop?**
A: See the demo script run output for this repo — record which model
successfully completed every topic (structured output for quiz
generation/grading/coaching, all via `with_structured_output`).
```

Before committing, fill in the last flashcard's answer with what Task 6
actually observed.

- [ ] **Step 2: Commit**

```bash
git add learning/chapter4.md
git commit -m "Add chapter 4 learning flashcards"
```

---

### Task 8: Open the chapter 4 PR

**Files:** none (branch/PR operation only)

- [ ] **Step 1: Confirm all tests pass**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Push the branch**

**Before running this step, ask the user to confirm.**

```bash
git push -u origin agent/chapter-4
```

- [ ] **Step 3: Open the PR**

**Before running this step, ask the user to confirm.**

```bash
gh pr create --title "Chapter 4: Building the Four-Agent System" --body "$(cat <<'EOF'
## Summary
- Extend shared state: QuizResult.weak_areas, PASS_THRESHOLD,
  get_current_topic(), session_is_complete()
- Add the Quiz Generator agent: question generation (temperature=0.4) and
  grading (temperature=0.1) as separate LLM calls, run_quiz() orchestration
  with an injectable answer_source (article uses real input() directly;
  this repo can't drive real stdin from its tooling, so the default still
  calls real input() but the demo swaps in canned answers)
- Add the Progress Coach agent: coaching message (temperature=0.4), topic
  status transitions, memory persistence, and route_after_coach() loop/end
  routing
- Wire both into the graph: curriculum_planner -> explainer ->
  quiz_generator -> progress_coach ->(conditional)-> explainer | END
- No human_approval node yet (that's chapter 5 per our spec, even though
  the source article's chapter 4 code already includes it)
- Add learning/chapter4.md flashcards

## Test plan
- [ ] `uv sync && uv run pytest -v` — all tests pass
- [ ] `uv run python scripts/demo_chapter4.py` — loops through every
      roadmap topic, ending with each marked completed or needs_review

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report the PR URL to the user**
