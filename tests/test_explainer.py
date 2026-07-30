from unittest.mock import MagicMock

from learning_accelerator.agents import explainer
from learning_accelerator.agents.explainer import MAX_ITERATIONS, explainer_node
from learning_accelerator.graph.state import StudyRoadmap, Topic, initial_state


def _make_state_with_topic():
    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=4,
        topics=[Topic(title="Intro", description="Basics", estimated_minutes=30)],
    )
    state = initial_state(goal="Learn LangGraph", session_id="s1")
    state["roadmap"] = roadmap
    return state


def test_explainer_node_returns_final_response_after_one_tool_round_trip(monkeypatch):
    state = _make_state_with_topic()

    tool_call_response = MagicMock()
    tool_call_response.tool_calls = [
        {"name": "tool_list_files", "args": {}, "id": "call_1"}
    ]

    final_response = MagicMock()
    final_response.tool_calls = []
    final_response.content = "LangGraph basics explained."

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [tool_call_response, final_response]
    fake_chat_model = MagicMock()
    fake_chat_model.bind_tools.return_value = fake_llm
    monkeypatch.setattr(
        explainer, "get_chat_model", MagicMock(return_value=fake_chat_model)
    )

    result = explainer_node(state)

    assert result == {"messages": [final_response], "error": None}
    assert fake_llm.invoke.call_count == 2


def test_explainer_node_returns_error_when_max_iterations_exceeded(monkeypatch):
    state = _make_state_with_topic()

    looping_response = MagicMock()
    looping_response.tool_calls = [
        {"name": "tool_list_files", "args": {}, "id": "call_loop"}
    ]

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = looping_response
    fake_chat_model = MagicMock()
    fake_chat_model.bind_tools.return_value = fake_llm
    monkeypatch.setattr(
        explainer, "get_chat_model", MagicMock(return_value=fake_chat_model)
    )

    result = explainer_node(state)

    assert result == {"error": "explainer exceeded max iterations"}
    assert fake_llm.invoke.call_count == MAX_ITERATIONS
