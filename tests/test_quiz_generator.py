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
    prior_result = QuizResult(topic="Setup", score=1.0, passed=True, weak_areas=[])
    state["quiz_results"] = [prior_result]

    quiz_result = QuizResult(
        topic="Intro", score=0.8, passed=True, weak_areas=["recursion"]
    )
    mock_run_quiz = MagicMock(return_value=quiz_result)
    monkeypatch.setattr(quiz_generator, "run_quiz", mock_run_quiz)

    result = quiz_generator_node(state)

    assert result["quiz_results"] == [prior_result, quiz_result]
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
