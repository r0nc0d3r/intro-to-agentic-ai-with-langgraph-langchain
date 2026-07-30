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
