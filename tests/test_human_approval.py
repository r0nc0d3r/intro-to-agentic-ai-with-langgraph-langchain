from learning_accelerator.agents.human_approval import route_after_approval
from learning_accelerator.graph.state import initial_state


def test_route_after_approval_continues_when_approved():
    state = initial_state(goal="g", session_id="s")
    state["approved"] = True

    assert route_after_approval(state) == "explainer"


def test_route_after_approval_regenerates_when_rejected():
    state = initial_state(goal="g", session_id="s")
    state["approved"] = False

    assert route_after_approval(state) == "curriculum_planner"


def test_route_after_approval_defaults_to_regenerate_when_missing():
    state = initial_state(goal="g", session_id="s")
    del state["approved"]

    assert route_after_approval(state) == "curriculum_planner"
