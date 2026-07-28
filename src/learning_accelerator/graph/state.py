from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

PASS_THRESHOLD = 0.5


class Topic(BaseModel):
    title: str
    description: str
    estimated_minutes: int
    prerequisites: list[str] = Field(default_factory=list)
    status: str = "pending"


class StudyRoadmap(BaseModel):
    goal: str
    total_weeks: int
    topics: list[Topic]
    weekly_hours: int = 5


@dataclass
class GradedAnswer:
    question: str
    expected_answer: str
    user_answer: str
    correct: bool
    feedback: str
    score: float


@dataclass
class QuizResult:
    topic: str
    score: float
    passed: bool
    weak_areas: list[str] = field(default_factory=list)
    questions: list[GradedAnswer] = field(default_factory=list)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    goal: str
    roadmap: Optional[StudyRoadmap]
    approved: bool
    current_topic_index: int
    quiz_results: list[QuizResult]
    weak_areas: list[str]
    study_materials_path: str
    error: Optional[str]


def initial_state(
    goal: str, session_id: str, study_materials_path: str = ""
) -> AgentState:
    return AgentState(
        messages=[],
        session_id=session_id,
        goal=goal,
        roadmap=None,
        approved=False,
        current_topic_index=0,
        quiz_results=[],
        weak_areas=[],
        study_materials_path=study_materials_path,
        error=None,
    )


def get_current_topic(state: AgentState) -> Topic:
    """Return the topic the graph is currently working through.

    Only valid while session_is_complete(state) is False.
    """
    return state["roadmap"].topics[state["current_topic_index"]]


def session_is_complete(state: AgentState) -> bool:
    roadmap = state.get("roadmap")
    if roadmap is None:
        return True
    return state.get("current_topic_index", 0) >= len(roadmap.topics)
