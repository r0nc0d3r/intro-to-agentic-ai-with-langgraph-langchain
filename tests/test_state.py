import pytest
from pydantic import ValidationError

from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state


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
