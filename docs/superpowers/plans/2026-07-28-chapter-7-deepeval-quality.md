# Chapter 7: Evaluating Agent Quality with DeepEval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated LLM-as-judge quality evaluation (DeepEval) for the
Explainer, Quiz Generator, and Progress Coach, gated behind a pytest `eval`
marker so these slow/non-deterministic tests stay out of the default fast
suite (currently 45 tests, 0 network calls).

**Source material note:** the article's own text for chapter 7 wasn't
retrievable from the page fetch, but the article references a companion
reference implementation repo
(`github.com/sandeepmb/freecodecamp-multi-agent-ai-system`) whose
`tests/test_eval.py` is the actual real source for this chapter — pulled
directly via `gh api` and used as the structural basis here, adapted to
this codebase's own conventions (see Deviations below).

**Architecture:** `evaluation/judge_model.py` provides
`LearningAcceleratorJudge(DeepEvalBaseLLM)` and `get_judge_model()`.
`tests/conftest.py` adds a fixture reading our own
`study_materials/sample_notes/langgraph_basics.md`. `tests/test_eval.py`
mirrors the reference repo's four test classes (Explainer quality, Quiz
Generator quality, Grading calibration, Progress Coach quality) using
`FaithfulnessMetric`, `AnswerRelevancyMetric`, and `GEval` from DeepEval,
all marked `@pytest.mark.eval`.

**Deliberate deviations from the reference repo (all judgment calls made
under the full autonomy already granted for chapters 5-9):**
1. **Judge model is provider-agnostic, not hardcoded to Ollama.** The
   reference's `OllamaJudge` wraps `ChatOllama` directly. This repo
   already has a provider abstraction (`config.get_chat_model()`,
   supporting ollama/anthropic/openai via `LLM_PROVIDER`) — the judge
   model reuses it instead of duplicating a second, Ollama-only path.
2. **Domain content is "LangGraph Basics", not "Python closures".** The
   reference repo's own study notes happen to be about Python
   programming topics; this repo's `study_materials/sample_notes/` are
   about LangGraph itself (`langgraph_basics.md`,
   `state_management.md`), matching this course's actual subject matter.
   All example questions/answers below are rewritten around that content.
3. **Attribute access, not dict access.** The reference's
   `generate_questions`/`grade_answer`/`get_coaching_message` return
   plain dicts (`q["question"]`, `result.get("score")`). This codebase's
   versions (built in chapter 4) return Pydantic models
   (`QuizQuestion`, `GradeResult`, `CoachingMessage`) — tests use
   `.question`/`.expected_answer`/`.score`/`.feedback`/`.summary`/`.tip`
   attribute access instead. Note: our `CoachingMessage` field is `tip`,
   not the reference's `encouragement` — that's this repo's own chapter 4
   naming (already merged, not renamed here).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-7` (already created, off updated `main`)
- One PR per chapter; push, open, and merge it yourself once verified —
  full autonomy already authorized for chapters 5-9
- `LearningAcceleratorJudge`'s construction, `load_model()`, and
  `get_model_name()` are testable without a live LLM call (constructing
  the judge and calling `get_chat_model()` doesn't make a network call,
  same as chapters 2/6's config tests) → pytest coverage, NOT marked
  `eval`. Its `generate()`/`a_generate()` methods make real calls and are
  only exercised by the `eval`-marked tests.
- ALL of `tests/test_eval.py` is `@pytest.mark.eval` — these tests call
  real agent functions (`explainer_node`, `generate_questions`,
  `grade_answer`, `get_coaching_message`) which make live LLM calls, so
  none of it qualifies as "pure logic" under this spec's testing policy
- `pyproject.toml` must register the `eval` marker AND set
  `addopts = "-m 'not eval'"` so plain `uv run pytest -v` (no `-m` flag)
  continues to run only the fast 45+ tests with zero network calls;
  `uv run pytest tests/test_eval.py -v -s -m eval` opts back in explicitly
- Thresholds start at 0.6 (matching the reference repo's own conservative
  choice for local models) — if a metric consistently fails at this
  threshold against `gemma4:12b`, the fix is checking the prompt/model
  first, not silently lowering the threshold (same principle the
  reference repo's own docstring states)
- Local dev machine has Ollama with `gemma4:12b-mlx`, `gemma4:12b`,
  `qwen3.5:2b`; chapters 2-6 all found `gemma4:12b` works reliably — use
  it as the default for both the agent-under-test AND the judge model
  here (same model, since there's no separate "stronger" model available
  locally — note this in the flashcards as a real limitation)
- `learning/chapterN.md` format: short Q/A flashcards (see chapters 1-6)

---

### Task 1: DeepEval judge model wrapper + tests

**Files:**
- Create: `src/learning_accelerator/evaluation/__init__.py`
- Create: `src/learning_accelerator/evaluation/judge_model.py`
- Test: `tests/test_judge_model.py`

**Interfaces:**
- Consumes: `get_chat_model` (chapter 2)
- Produces: `LearningAcceleratorJudge` (a `DeepEvalBaseLLM` subclass),
  `get_judge_model() -> LearningAcceleratorJudge` — used by
  `tests/test_eval.py` in Task 4

- [ ] **Step 1: Add the deepeval dependency**

```bash
uv add deepeval
```

Expected: `pyproject.toml`/`uv.lock` updated.

- [ ] **Step 2: Verify the exact import paths available in the installed version**

Run:
```bash
uv run python -c "from deepeval.models import DeepEvalBaseLLM; print('ok: deepeval.models')"
uv run python -c "from deepeval.test_case import LLMTestCaseParams; print('ok: LLMTestCaseParams')"
```

If either import fails, find the correct current path/name in the
installed package (e.g. `deepeval.models.base_model`, or
`SingleTurnParams` as an alternate/renamed export for the second import)
and use whatever actually works in place of the exact names below — note
in your report which path you used and why.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_judge_model.py`:

```python
from learning_accelerator.evaluation.judge_model import (
    LearningAcceleratorJudge,
    get_judge_model,
)


def test_get_judge_model_returns_judge_instance():
    judge = get_judge_model()
    assert isinstance(judge, LearningAcceleratorJudge)


def test_judge_model_name_includes_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    judge = get_judge_model()
    assert "ollama" in judge.get_model_name()


def test_judge_load_model_uses_configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:12b")
    judge = get_judge_model()
    model = judge.load_model()
    assert type(model).__name__ == "ChatOllama"
    assert model.temperature == 0.0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_judge_model.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.evaluation'`

- [ ] **Step 5: Create `src/learning_accelerator/evaluation/__init__.py`**

```python
```

(empty file)

- [ ] **Step 6: Write `src/learning_accelerator/evaluation/judge_model.py`**

```python
from __future__ import annotations

import os

from deepeval.models import DeepEvalBaseLLM

from learning_accelerator.config import get_chat_model


class LearningAcceleratorJudge(DeepEvalBaseLLM):
    """DeepEval judge model backed by this project's own provider-agnostic
    get_chat_model() — follows the same LLM_PROVIDER env var as the rest
    of the app (ollama/anthropic/openai), rather than being hardcoded to
    a single provider.
    """

    def load_model(self):
        return get_chat_model(temperature=0.0)

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        provider = os.environ.get("LLM_PROVIDER", "ollama")
        return f"learning-accelerator-judge/{provider}"


def get_judge_model() -> LearningAcceleratorJudge:
    return LearningAcceleratorJudge()
```

(Adjust the `DeepEvalBaseLLM` import path if Step 2 found a different
one works in the installed version.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_judge_model.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/learning_accelerator/evaluation/__init__.py src/learning_accelerator/evaluation/judge_model.py tests/test_judge_model.py
git commit -m "Add provider-agnostic DeepEval judge model wrapper"
```

---

### Task 2: pytest eval marker configuration

**Files:**
- Modify: `pyproject.toml`

**Interfaces:** none (config only)

- [ ] **Step 1: Add a `[tool.pytest.ini_options]` section**

`pyproject.toml` currently has no `[tool.pytest.ini_options]` section at
all (pytest reports `configfile: pyproject.toml` merely because the file
exists at the project root, not because any settings are defined there).
Add this section (placement anywhere in the file is fine, e.g. after
`[dependency-groups]`):

```toml
[tool.pytest.ini_options]
markers = [
    "eval: slow, non-deterministic LLM-as-judge evaluation tests (needs a live LLM_PROVIDER) — excluded from the default run",
]
addopts = "-m 'not eval'"
```

- [ ] **Step 2: Verify the default run still excludes nothing unexpected**

Run: `uv run pytest -v`
Expected: the same 48 tests from Task 1 pass (45 pre-existing + 3 new
judge_model tests), 0 skipped, 0 deselected — because no test in the
repo carries the `eval` marker yet (Task 4 adds those).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Add pytest eval marker, excluded from the default test run"
```

---

### Task 3: Shared fixture for study note content

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `langgraph_basics_note_content` fixture — used by
  `tests/test_eval.py` in Task 4

- [ ] **Step 1: Write `tests/conftest.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

NOTES_DIR = Path(__file__).parent.parent / "study_materials" / "sample_notes"


@pytest.fixture
def langgraph_basics_note_content() -> str:
    return (NOTES_DIR / "langgraph_basics.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Verify it doesn't break existing test collection**

Run: `uv run pytest -v`
Expected: same 48 tests pass as Task 2 (a bare `conftest.py` with one
unused-by-existing-tests fixture doesn't affect anything yet).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "Add shared fixture for study note content in tests"
```

---

### Task 4: DeepEval quality test suite

**Files:**
- Create: `tests/test_eval.py`

**Interfaces:**
- Consumes: `get_judge_model` (Task 1), `langgraph_basics_note_content`
  fixture (Task 3), `explainer_node`/`StudyRoadmap`/`Topic`/`initial_state`
  (chapters 2-3), `generate_questions`/`grade_answer` (chapter 4),
  `get_coaching_message` (chapter 4)

This is the chapter's actual deliverable — not a case of writing a
failing test then production code, since these tests exercise
already-built agents. Write the whole file, then run it for real (Step 2)
and treat any assertion failure at the stated thresholds as a real signal
to investigate (adjust the *test's* prompt/expected-answer text if it's
poorly worded, or the threshold if genuinely too strict for a local 12B
model — document whichever you do and why in your report).

- [ ] **Step 1: Write `tests/test_eval.py`**

```python
"""
tests/test_eval.py

LLM-as-judge evaluation tests for the Learning Accelerator.

TIER 2 TESTS: require a live LLM_PROVIDER (Ollama by default). These
tests are slow (30-120s each) and non-deterministic. Excluded from the
default `pytest` run via the "eval" marker (see pyproject.toml) — run
them explicitly:

  uv run pytest tests/test_eval.py -v -s -m eval

What these tests check:
  - Explainer explanations are faithful to source notes
  - Explainer explanations are relevant to the question asked
  - Quiz questions test understanding, not just recall
  - The grader scores correct/wrong/partial answers sensibly
  - Progress Coach messages are specific, not generic

Thresholds are set conservatively (0.6) to account for variability in
local model outputs. If a test consistently fails, check the model and
prompt first before lowering the threshold.
"""

from __future__ import annotations

import pytest


def _run_explainer(topic_title: str, topic_description: str, session_id: str) -> str:
    from langchain_core.messages import AIMessage

    from learning_accelerator.agents.explainer import explainer_node
    from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state

    state = initial_state(goal=f"Learn {topic_title}", session_id=session_id)
    state["roadmap"] = StudyRoadmap(
        goal=f"Learn {topic_title}",
        total_weeks=1,
        topics=[
            Topic(title=topic_title, description=topic_description, estimated_minutes=60)
        ],
    )
    state["current_topic_index"] = 0

    result = explainer_node(state)

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""


@pytest.mark.eval
class TestExplainerQuality:
    FAITHFULNESS_THRESHOLD = 0.6
    RELEVANCY_THRESHOLD = 0.6

    @pytest.fixture(autouse=True)
    def setup(self, langgraph_basics_note_content):
        self.retrieval_context = [langgraph_basics_note_content]
        print("\n[TestExplainerQuality] Running Explainer for LangGraph Basics...")
        self.explanation = _run_explainer(
            topic_title="LangGraph Basics",
            topic_description="Understand nodes, edges, state, and checkpointing in LangGraph",
            session_id="eval-test-001",
        )
        if not self.explanation:
            pytest.skip("Explainer returned empty output — check the configured LLM provider is reachable")
        print(f"[TestExplainerQuality] Explanation length: {len(self.explanation)} chars")

    def test_explanation_is_faithful_to_notes(self):
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        from learning_accelerator.evaluation.judge_model import get_judge_model

        test_case = LLMTestCase(
            input="Explain LangGraph basics",
            actual_output=self.explanation,
            retrieval_context=self.retrieval_context,
        )
        metric = FaithfulnessMetric(
            model=get_judge_model(),
            threshold=self.FAITHFULNESS_THRESHOLD,
            include_reason=True,
        )
        metric.measure(test_case)

        print(f"\n[Faithfulness] Score: {metric.score:.3f} (threshold: {self.FAITHFULNESS_THRESHOLD})")
        if metric.reason:
            print(f"[Faithfulness] Reason: {metric.reason}")

        assert metric.score >= self.FAITHFULNESS_THRESHOLD, (
            f"Faithfulness score {metric.score:.3f} below threshold {self.FAITHFULNESS_THRESHOLD}.\n"
            f"Reason: {metric.reason}"
        )

    def test_explanation_is_relevant_to_topic(self):
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        from learning_accelerator.evaluation.judge_model import get_judge_model

        test_case = LLMTestCase(
            input="Explain LangGraph basics: nodes, edges, state, and checkpointing",
            actual_output=self.explanation,
        )
        metric = AnswerRelevancyMetric(
            model=get_judge_model(),
            threshold=self.RELEVANCY_THRESHOLD,
            include_reason=True,
        )
        metric.measure(test_case)

        print(f"\n[Relevancy] Score: {metric.score:.3f} (threshold: {self.RELEVANCY_THRESHOLD})")

        assert metric.score >= self.RELEVANCY_THRESHOLD, (
            f"Relevancy score {metric.score:.3f} below threshold {self.RELEVANCY_THRESHOLD}."
        )

    def test_explanation_has_minimum_length(self):
        min_length = 150
        assert len(self.explanation) >= min_length, (
            f"Explanation too short: {len(self.explanation)} chars (minimum: {min_length}).\n"
            f"Content: {self.explanation[:200]}"
        )

    def test_explanation_mentions_key_concepts(self):
        keywords = ["node", "edge", "state", "checkpoint", "graph", "reducer"]
        explanation_lower = self.explanation.lower()
        found = [kw for kw in keywords if kw in explanation_lower]
        assert len(found) >= 2, (
            "Explanation mentions too few LangGraph concepts.\n"
            f"Found: {found}\nExpected at least 2 of: {keywords}\n"
            f"Explanation preview: {self.explanation[:300]}"
        )


@pytest.mark.eval
class TestQuizGeneratorQuality:
    QUESTION_QUALITY_THRESHOLD = 0.6

    def test_generated_questions_test_understanding(self, langgraph_basics_note_content):
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        from learning_accelerator.agents.quiz_generator import generate_questions
        from learning_accelerator.evaluation.judge_model import get_judge_model

        print("\n[TestQuizQuality] Generating quiz questions...")
        questions = generate_questions(
            topic="LangGraph Basics", explanation=langgraph_basics_note_content, n=3
        )
        assert len(questions) > 0, "No questions were generated"
        print(f"[TestQuizQuality] Generated {len(questions)} questions")

        questions_text = "\n".join(
            f"Q{i + 1}: {q.question}\nExpected: {q.expected_answer}"
            for i, q in enumerate(questions)
        )

        test_case = LLMTestCase(
            input="Generate quiz questions about LangGraph basics that test understanding",
            actual_output=questions_text,
        )
        metric = GEval(
            name="QuestionQuality",
            criteria=(
                "Evaluate whether these quiz questions test genuine conceptual "
                "understanding of LangGraph rather than surface-level recall. Good "
                "questions require the student to apply concepts, explain WHY "
                "something works, identify edge cases, or compare approaches. Poor "
                "questions only ask to define terms or recite the notes verbatim."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=get_judge_model(),
            threshold=self.QUESTION_QUALITY_THRESHOLD,
        )
        metric.measure(test_case)

        print(f"\n[QuestionQuality] Score: {metric.score:.3f} (threshold: {self.QUESTION_QUALITY_THRESHOLD})")
        if metric.reason:
            print(f"[QuestionQuality] Reason: {metric.reason}")

        assert metric.score >= self.QUESTION_QUALITY_THRESHOLD, (
            f"Question quality score {metric.score:.3f} below threshold.\n"
            f"Questions generated:\n{questions_text}"
        )

    def test_questions_have_required_structure(self, langgraph_basics_note_content):
        from learning_accelerator.agents.quiz_generator import generate_questions

        questions = generate_questions(
            topic="LangGraph Basics", explanation=langgraph_basics_note_content, n=3
        )

        assert isinstance(questions, list)
        assert len(questions) > 0

        for i, q in enumerate(questions):
            assert len(q.question) > 10, f"Question {i} text too short"
            assert len(q.expected_answer) > 10, f"Question {i} answer too short"
            assert q.question.strip().endswith("?"), (
                f"Question {i} should end with '?': {q.question}"
            )


@pytest.mark.eval
class TestGradingQuality:
    def test_correct_answer_scores_high(self):
        from learning_accelerator.agents.quiz_generator import grade_answer

        question = "What are the two ways a LangGraph state field can update?"
        expected = (
            "Last-write-wins (a node's returned value simply replaces the "
            "previous one) or reducer-based (a function like add_messages "
            "controls how new values combine with old ones)."
        )
        student_answer = (
            "A field either gets overwritten by whatever the node returns "
            "(last-write-wins), or it uses a reducer function like "
            "add_messages that decides how to merge the new value with "
            "what's already there."
        )

        result = grade_answer(question, expected, student_answer)
        print(f"\n[GradeQuality] Correct answer score: {result.score:.2f}")
        print(f"[GradeQuality] Feedback: {result.feedback}")

        assert result.score >= 0.65, (
            f"Correct answer scored too low: {result.score:.2f}.\nFeedback: {result.feedback}"
        )

    def test_wrong_answer_scores_low(self):
        from learning_accelerator.agents.quiz_generator import grade_answer

        question = "What is a LangGraph checkpoint?"
        expected = (
            "A saved snapshot of the graph's state after a step, persisted "
            "so a run can be paused, resumed, or recovered after a crash."
        )
        student_answer = (
            "A checkpoint is a function that checks whether the LLM's "
            "output is valid before letting the graph continue."
        )

        result = grade_answer(question, expected, student_answer)
        print(f"\n[GradeQuality] Wrong answer score: {result.score:.2f}")
        print(f"[GradeQuality] Feedback: {result.feedback}")

        assert result.score <= 0.35, (
            f"Wrong answer scored too high: {result.score:.2f}.\nFeedback: {result.feedback}"
        )

    def test_partial_answer_scores_middle(self):
        from learning_accelerator.agents.quiz_generator import grade_answer

        question = (
            "Why must SqliteSaver be constructed from a raw connection "
            "instead of a context manager?"
        )
        expected = (
            "LangGraph runs node functions and checkpoint writes on "
            "different threads, so the connection needs "
            "check_same_thread=False, and it must stay open for the whole "
            "process — a `with` block would close it too early."
        )
        student_answer = (
            "Because the connection needs to stay open the whole time the "
            "app is running, not just for one function call."
        )

        result = grade_answer(question, expected, student_answer)
        print(f"\n[GradeQuality] Partial answer score: {result.score:.2f}")
        print(f"[GradeQuality] Feedback: {result.feedback}")

        assert 0.3 <= result.score <= 0.8, (
            f"Partial answer should score in a middle range, got {result.score:.2f}.\n"
            f"Feedback: {result.feedback}"
        )

    def test_grader_returns_valid_score_range(self):
        from learning_accelerator.agents.quiz_generator import grade_answer

        result = grade_answer(
            "What is a closure?",
            "A nested function capturing outer variables.",
            "Some student answer",
        )
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0


@pytest.mark.eval
class TestProgressCoachQuality:
    COACHING_QUALITY_THRESHOLD = 0.6

    def test_coaching_message_is_specific_not_generic(self):
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        from learning_accelerator.agents.progress_coach import get_coaching_message
        from learning_accelerator.evaluation.judge_model import get_judge_model

        print("\n[CoachQuality] Generating coaching message...")
        coaching = get_coaching_message(
            topic="LangGraph Basics", score=0.67, weak_areas=["checkpointing", "reducers"]
        )
        coaching_text = f"Summary: {coaching.summary}\nTip: {coaching.tip}"
        print(f"[CoachQuality] Message:\n{coaching_text}")

        test_case = LLMTestCase(
            input=(
                "Generate coaching feedback for a student who scored 67% on "
                "LangGraph Basics and struggled with checkpointing and reducers"
            ),
            actual_output=coaching_text,
        )
        metric = GEval(
            name="CoachingQuality",
            criteria=(
                "Evaluate whether this coaching message is: 1) encouraging "
                "without being dishonest about the score, 2) specific to "
                "the topic and weak areas mentioned, 3) actionable, gives "
                "the student a clear next step, 4) appropriately concise. "
                "A poor message is generic, vague, or condescending."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=get_judge_model(),
            threshold=self.COACHING_QUALITY_THRESHOLD,
        )
        metric.measure(test_case)

        print(f"\n[CoachingQuality] Score: {metric.score:.3f} (threshold: {self.COACHING_QUALITY_THRESHOLD})")

        assert metric.score >= self.COACHING_QUALITY_THRESHOLD, (
            f"Coaching quality {metric.score:.3f} below threshold.\nMessage:\n{coaching_text}"
        )

    def test_coaching_returns_required_fields(self):
        from learning_accelerator.agents.progress_coach import get_coaching_message

        result = get_coaching_message(topic="Test Topic", score=0.8, weak_areas=[])

        assert isinstance(result.summary, str) and len(result.summary) > 0
        assert isinstance(result.tip, str) and len(result.tip) > 0
```

- [ ] **Step 2: Run the eval suite for real against local Ollama**

This will be slow (the reference repo's own estimate is 30-120s per
test, and there are 10 test functions here — expect 10-20+ minutes
total). Run in the background rather than blocking.

```bash
rm -f .data/checkpoints.sqlite
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b uv run pytest tests/test_eval.py -v -s -m eval
```

Expected: all tests pass. If any metric-based test fails at the stated
threshold against `gemma4:12b`, read the printed `Reason`/`Feedback`
output — if the test's own prompt or expected-answer wording is
ambiguous or unfair, fix the test; if the model's actual output is
genuinely weak in a way a stronger judge/model would also flag, that's a
legitimate finding for the chapter 7 flashcards (this chapter's whole
point is surfacing exactly this kind of signal), not something to
silently paper over by lowering the threshold without comment.

- [ ] **Step 3: Confirm the default fast suite is still unaffected**

Run: `uv run pytest -v`
Expected: same 48 tests as Task 3 (still 0 eval-marked tests running by
default).

- [ ] **Step 4: Commit**

```bash
git add tests/test_eval.py
git commit -m "Add DeepEval LLM-as-judge quality tests for Explainer, Quiz Generator, Progress Coach"
```

---

### Task 5: Chapter 7 learning flashcards

**Files:**
- Create: `learning/chapter7.md`

- [ ] **Step 1: Write `learning/chapter7.md`**

Cover at minimum: what DeepEval metrics were used and why
(FaithfulnessMetric, AnswerRelevancyMetric, GEval), why a custom judge
model was needed and how it reuses this repo's provider abstraction, why
these tests are marked `eval` and excluded from the default run, and the
actual observed results from Task 4's real run (which metric(s) passed
cleanly, any that needed adjustment, and — an honest limitation worth
recording — that the judge model and the agent-under-test are currently
the *same* local model (`gemma4:12b`), which is weaker than the
best-practice of using a stronger/different model as judge to avoid the
model grading its own homework.

Fill in the real observed scores/behavior from Task 4 before committing
— don't leave placeholder text.

- [ ] **Step 2: Commit**

```bash
git add learning/chapter7.md
git commit -m "Add chapter 7 learning flashcards"
```

---

### Task 6: Push, open, and merge the chapter 7 PR

**Files:** none (branch/PR operation only)

Full autonomy is authorized for chapters 5-9 — no confirmation needed
before push/PR/merge. Before this task, dispatch a final whole-branch
review (per subagent-driven-development) covering all of Tasks 1-5
together, and address any Critical/Important findings before proceeding.

- [ ] **Step 1: Confirm the default fast suite passes**

Run: `uv run pytest -v`

- [ ] **Step 2: Push the branch**

```bash
git push -u origin agent/chapter-7
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "Chapter 7: Evaluating Agent Quality with DeepEval" --body "..."
```

(Compose the body summarizing what was built and the real eval results,
plus a test plan section — same style as chapters 1-6's PRs.)

- [ ] **Step 4: Merge**

```bash
gh pr merge --merge
```

- [ ] **Step 5: Delete the remote branch and sync the local worktree**

```bash
git push origin --delete agent/chapter-7
git checkout claude/agentic-ai-langgraph-course-ebf06b
git fetch origin
git merge --ff-only origin/main
git branch -d agent/chapter-7
```
