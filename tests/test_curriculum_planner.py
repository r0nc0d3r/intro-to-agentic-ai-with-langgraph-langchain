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
