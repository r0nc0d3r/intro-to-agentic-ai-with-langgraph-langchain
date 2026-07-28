from __future__ import annotations

from langgraph.types import interrupt

from learning_accelerator.graph.state import AgentState


def _parse_approval(decision: object) -> bool:
    """Interpret a resumed approval decision as a bool.

    Handles the natural bool case explicitly (e.g. a UI's "Approve" button
    sending ``Command(resume=True)``) before falling back to string
    matching — otherwise ``str(True).lower()`` == ``"true"``, which isn't
    in the approval-word tuple, and a real approval gets silently treated
    as a rejection.
    """
    if isinstance(decision, bool):
        return decision
    return str(decision).strip().lower() in ("yes", "y", "ok", "approve", "true")


def human_approval_node(state: AgentState) -> dict:
    roadmap = state.get("roadmap")

    decision = interrupt(
        {
            "type": "roadmap_approval",
            "roadmap": roadmap,
            "prompt": "Does this study plan look good? (yes/no)",
        }
    )

    approved = _parse_approval(decision)

    return {"approved": approved, "error": None}


def route_after_approval(state: AgentState) -> str:
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"
