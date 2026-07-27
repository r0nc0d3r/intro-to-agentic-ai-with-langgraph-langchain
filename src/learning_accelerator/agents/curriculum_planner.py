from __future__ import annotations

from langchain_core.messages import AIMessage

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import AgentState, StudyRoadmap

SYSTEM_PROMPT = """You are a curriculum planner. Given a learning goal, \
produce a study roadmap.

Rules:
- Produce 4 to 6 topics, ordered from foundational to advanced.
- Every topic's "prerequisites" must exactly match an earlier topic's title.
- Every topic's "status" is always "pending".
"""


def curriculum_planner_node(state: AgentState) -> dict:
    llm = get_chat_model(temperature=0.1).with_structured_output(StudyRoadmap)

    roadmap = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Learning goal: {state['goal']}"},
        ]
    )

    summary = AIMessage(
        content=(
            f"Planned {len(roadmap.topics)} topics over "
            f"{roadmap.total_weeks} weeks."
        )
    )

    return {
        "roadmap": roadmap,
        "messages": [summary],
        "error": None,
    }
