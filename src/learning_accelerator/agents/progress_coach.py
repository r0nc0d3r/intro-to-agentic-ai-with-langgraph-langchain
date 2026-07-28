from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import AgentState, PASS_THRESHOLD, session_is_complete
from learning_accelerator.mcp_servers.memory_server import memory_set

COACHING_PROMPT = """You are a warm, encouraging study coach. Given a topic, \
a score, and any weak areas, write a short encouraging summary and one \
concrete tip for what to review next.
"""


class CoachingMessage(BaseModel):
    summary: str
    tip: str


def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> CoachingMessage:
    llm = get_chat_model(temperature=0.4).with_structured_output(CoachingMessage)
    context = {
        "topic": topic,
        "score_percent": f"{score:.0%}",
        "weak_areas": weak_areas if weak_areas else ["none identified"],
    }
    return llm.invoke(
        [
            SystemMessage(content=COACHING_PROMPT),
            HumanMessage(content=json.dumps(context)),
        ]
    )


def next_topic_status(score: float) -> str:
    return "completed" if score >= PASS_THRESHOLD else "needs_review"


def progress_coach_node(state: AgentState) -> dict:
    quiz_results = state.get("quiz_results", [])
    latest = quiz_results[-1]
    roadmap = state["roadmap"]
    idx = state.get("current_topic_index", 0)
    session_id = state["session_id"]

    coaching = get_coaching_message(latest.topic, latest.score, latest.weak_areas)

    if idx < len(roadmap.topics):
        roadmap.topics[idx].status = next_topic_status(latest.score)

    memory_set(
        session_id,
        f"progress_topic_{idx}",
        json.dumps(
            {
                "topic": latest.topic,
                "score": latest.score,
                "weak_areas": latest.weak_areas,
            }
        ),
    )

    return {
        "roadmap": roadmap,
        "current_topic_index": idx + 1,
        "messages": [AIMessage(content=coaching.summary)],
        "error": None,
    }


def route_after_coach(state: AgentState) -> str:
    return "end" if session_is_complete(state) else "explainer"
