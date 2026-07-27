from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


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
class QuizResult:
    topic: str
    score: float
    passed: bool


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
