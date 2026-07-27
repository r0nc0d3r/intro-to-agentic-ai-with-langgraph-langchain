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
