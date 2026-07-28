from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import (
    PASS_THRESHOLD,
    AgentState,
    QuizResult,
    get_current_topic,
)

GENERATION_PROMPT = """You are a quiz designer for a student learning \
programming. Given a topic and explanation, generate quiz questions that \
test genuine understanding, not rote recall.

Rules:
- Each question must end with "?".
- "expected_answer" is a model answer in 1-3 sentences.
- "difficulty" is one of "easy", "medium", "hard".
"""

GRADING_PROMPT = """You are a fair teacher grading a student's answer. \
Compare the student's answer to the model answer and grade honestly — \
partial credit is fine.

Rules:
- "score" is a float between 0.0 and 1.0.
- "correct" is true only if the score is 1.0.
- "feedback" is one specific sentence.
- "missing_concept" is the key concept the student missed, or "" if the
  answer is correct.
"""


class QuizQuestion(BaseModel):
    question: str
    expected_answer: str
    difficulty: str = "medium"


class QuestionSet(BaseModel):
    questions: list[QuizQuestion]


class GradeResult(BaseModel):
    correct: bool
    score: float
    feedback: str
    missing_concept: str = ""


AnswerSource = Callable[[str], str]


def _default_answer_source(question: str) -> str:
    return input(f"{question}\nYour answer: ").strip()


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[QuizQuestion]:
    llm = get_chat_model(temperature=0.4).with_structured_output(QuestionSet)
    result = llm.invoke(
        [
            {"role": "system", "content": GENERATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\nExplanation: {explanation}\n"
                    f"Generate exactly {n} questions."
                ),
            },
        ]
    )
    return result.questions


def grade_answer(question: str, expected: str, student_answer: str) -> GradeResult:
    llm = get_chat_model(temperature=0.1).with_structured_output(GradeResult)
    return llm.invoke(
        [
            {"role": "system", "content": GRADING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nModel answer: {expected}\n"
                    f"Student's answer: {student_answer}"
                ),
            },
        ]
    )


def run_quiz(
    topic: str,
    explanation: str,
    n: int = 3,
    answer_source: AnswerSource | None = None,
) -> QuizResult:
    ask = answer_source or _default_answer_source
    questions = generate_questions(topic, explanation, n=n)

    total_score = 0.0
    weak_areas: list[str] = []

    for q in questions:
        student_answer = ask(q.question)
        grade = grade_answer(q.question, q.expected_answer, student_answer)
        total_score += grade.score
        if grade.missing_concept:
            weak_areas.append(grade.missing_concept)

    avg_score = total_score / len(questions) if questions else 0.0

    return QuizResult(
        topic=topic,
        score=avg_score,
        passed=avg_score >= PASS_THRESHOLD,
        weak_areas=weak_areas,
    )


def quiz_generator_node(state: AgentState) -> dict:
    topic = get_current_topic(state)

    explanation = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            explanation = msg.content
            break

    quiz_result = run_quiz(topic.title, explanation)

    all_weak_areas = list(set(state.get("weak_areas", []) + quiz_result.weak_areas))

    return {
        "quiz_results": state.get("quiz_results", []) + [quiz_result],
        "weak_areas": all_weak_areas,
        "error": None,
    }
