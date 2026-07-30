from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import (
    AgentState,
    PASS_THRESHOLD,
    get_last_explanation,
    session_is_complete,
)
from learning_accelerator.mcp_servers.memory_server import memory_set

COACHING_PROMPT = """You are a warm, encouraging study coach. Given a topic, \
a score, and any weak areas, write a short encouraging summary and one \
concrete tip for what to review next.
"""

QUIZ_SERVICE_URL = "http://localhost:9001"
STUDY_BUDDY_URL = "http://localhost:9002"


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


def try_a2a_quiz_delegation(
    topic: str, explanation: str, answers: list[str]
) -> dict | None:
    """Attempt to delegate quiz grading to the A2A Quiz Service.

    Returns the grading result dict if successful, None if the service is
    disabled, unavailable, or returns an error — callers should fall back
    to local quiz generation in that case. Not currently called by
    progress_coach_node (see learning/chapter8.md for why); available for
    direct/manual use.
    """
    use_a2a = os.environ.get("USE_A2A_QUIZ", "true").lower() == "true"
    if not use_a2a:
        return None

    from learning_accelerator.a2a_services.a2a_client import (
        delegate_quiz_task,
        is_quiz_service_available,
    )

    quiz_service_url = os.environ.get("QUIZ_SERVICE_URL", QUIZ_SERVICE_URL)

    if not is_quiz_service_available(quiz_service_url):
        print(
            f"[Progress Coach] Quiz A2A service not available at "
            f"{quiz_service_url}, using local quiz generator"
        )
        return None

    print(f"[Progress Coach] Delegating quiz to A2A service: {quiz_service_url}")
    result = delegate_quiz_task(
        topic=topic,
        explanation=explanation,
        answers=answers,
        quiz_service_url=quiz_service_url,
    )

    if "error" in result:
        print(f"[Progress Coach] A2A delegation failed: {result['error']}")
        return None

    print(f"[Progress Coach] A2A quiz complete: score={result.get('score', 0):.0%}")
    return result


def try_study_buddy_assistance(
    topic: str, explanation: str, weak_areas: list[str]
) -> str | None:
    """Request supplementary study help from the CrewAI Study Buddy.

    Called when a student scores below the pass threshold. Returns the
    assistance text if available, None if the service is disabled,
    unavailable, or returns an error.
    """
    use_study_buddy = os.environ.get("USE_STUDY_BUDDY", "true").lower() == "true"
    if not use_study_buddy:
        return None

    from learning_accelerator.a2a_services.a2a_client import (
        is_study_buddy_available,
        request_study_assistance,
    )

    study_buddy_url = os.environ.get("STUDY_BUDDY_URL", STUDY_BUDDY_URL)

    if not is_study_buddy_available(study_buddy_url):
        return None

    print("[Progress Coach] Requesting study assistance from CrewAI Study Buddy...")
    result = request_study_assistance(
        topic=topic,
        explanation=explanation,
        weak_areas=weak_areas,
        study_buddy_url=study_buddy_url,
    )

    if "error" in result or result.get("status") == "error":
        return None

    return result.get("assistance", "")


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

    if latest.score < PASS_THRESHOLD and latest.weak_areas:
        explanation = get_last_explanation(state)

        assistance = try_study_buddy_assistance(
            topic=latest.topic,
            explanation=explanation,
            weak_areas=latest.weak_areas,
        )
        if assistance:
            print("\n" + "─" * 60)
            print("Study Buddy (via CrewAI → A2A):")
            print(assistance)
            print("─" * 60 + "\n")

    return {
        "roadmap": roadmap,
        "current_topic_index": idx + 1,
        "messages": [AIMessage(content=coaching.summary)],
        "error": None,
    }


def route_after_coach(state: AgentState) -> str:
    return "end" if session_is_complete(state) else "explainer"
