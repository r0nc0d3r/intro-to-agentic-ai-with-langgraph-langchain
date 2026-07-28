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
