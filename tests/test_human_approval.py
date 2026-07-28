from learning_accelerator.agents.human_approval import (
    _parse_approval,
    route_after_approval,
)
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


def test_parse_approval_bool_true_is_approved():
    assert _parse_approval(True) is True


def test_parse_approval_bool_false_is_rejected():
    assert _parse_approval(False) is False


def test_parse_approval_yes_variants_are_approved():
    for value in ("yes", "y", "ok", "approve", "Yes", " YES ", "OK"):
        assert _parse_approval(value) is True


def test_parse_approval_no_variants_are_rejected():
    for value in ("no", "nope", ""):
        assert _parse_approval(value) is False
