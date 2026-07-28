from __future__ import annotations

from langgraph.types import interrupt

from learning_accelerator.graph.state import AgentState


def human_approval_node(state: AgentState) -> dict:
    roadmap = state.get("roadmap")

    decision = interrupt(
        {
            "type": "roadmap_approval",
            "roadmap": roadmap,
            "prompt": "Does this study plan look good? (yes/no)",
        }
    )

    approved = str(decision).strip().lower() in ("yes", "y", "ok", "approve")

    return {"approved": approved, "error": None}


def route_after_approval(state: AgentState) -> str:
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"
