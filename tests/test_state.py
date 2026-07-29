import pytest
from pydantic import ValidationError

from learning_accelerator.graph.state import (
    GradedAnswer,
    QuizResult,
    StudyRoadmap,
    Topic,
    get_current_topic,
    initial_state,
    session_is_complete,
)


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
