"""
Quiz Generator exposed as a standalone A2A service.

Run standalone:
  uv run python src/learning_accelerator/a2a_services/quiz_service.py

Then discover:
  curl http://localhost:9001/.well-known/agent-card.json
"""

from __future__ import annotations

import asyncio
import json
import uuid

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, TextPart

from learning_accelerator.agents.quiz_generator import generate_questions, grade_answer

QUIZ_SKILL = AgentSkill(
    id="generate_and_grade_quiz",
    name="Generate and Grade Quiz",
    description=(
        "Given a topic and optional explanation text, generates quiz questions "
        "that test conceptual understanding. If answers are provided, grades "
        "each answer and returns scores with identified weak areas."
    ),
    tags=["quiz", "assessment", "education", "grading"],
    examples=[
        "Generate a quiz on LangGraph state management",
        "Grade these answers for a checkpointing quiz: ...",
    ],
)

QUIZ_AGENT_CARD = AgentCard(
    name="Quiz Generator Service",
    description=(
        "A specialised quiz generation and grading service built with LangGraph. "
        "Generates questions that test genuine understanding, grades answers "
        "using LLM-as-judge, and identifies weak areas for further study. "
        "Framework-agnostic: works with any A2A-compatible agent."
    ),
    url="http://localhost:9001/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[QUIZ_SKILL],
)


class QuizAgentExecutor(AgentExecutor):
    """Handles incoming A2A quiz tasks.

    Request format (JSON in the text part):
    {"topic": "...", "explanation": "...", "answers": [...]}   (answers optional)

    Response format (JSON in the text part):
    {"status": "questions_ready" | "graded", "topic": ..., "questions": [...],
     "score": ..., "graded_questions": [...], "weak_areas": [...]}
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = context.get_user_input()

        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError:
            request_data = {"topic": request_text, "explanation": ""}

        topic = request_data.get("topic", "General Knowledge")
        explanation = request_data.get("explanation", "")
        provided_answers = request_data.get("answers", [])

        print(
            f"[Quiz A2A] Task received: topic='{topic}', "
            f"answers_provided={len(provided_answers)}"
        )

        questions = await asyncio.to_thread(generate_questions, topic, explanation, 3)
        questions_data = [q.model_dump() for q in questions]

        if not provided_answers:
            result = {
                "status": "questions_ready",
                "topic": topic,
                "questions": questions_data,
                "message": "Questions generated. Submit again with 'answers' key to grade.",
            }
        else:
            graded = []
            total_score = 0.0
            weak_areas: list[str] = []

            for q, answer in zip(questions, provided_answers):
                grade = await asyncio.to_thread(
                    grade_answer, q.question, q.expected_answer, answer
                )
                total_score += grade.score
                if grade.missing_concept:
                    weak_areas.append(grade.missing_concept)

                graded.append(
                    {
                        "question": q.question,
                        "answer": answer,
                        "score": grade.score,
                        "correct": grade.correct,
                        "feedback": grade.feedback,
                    }
                )

            avg_score = total_score / len(questions) if questions else 0.0

            result = {
                "status": "graded",
                "topic": topic,
                "score": avg_score,
                "questions": questions_data,
                "graded_questions": graded,
                "weak_areas": list(set(weak_areas)),
            }

        print(f"[Quiz A2A] Task complete: status={result['status']}")

        await event_queue.enqueue_event(
            Message(
                role="agent",
                message_id=str(uuid.uuid4()),
                parts=[TextPart(text=json.dumps(result, indent=2))],
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def create_quiz_server():
    request_handler = DefaultRequestHandler(
        agent_executor=QuizAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=QUIZ_AGENT_CARD, http_handler=request_handler)
    return app.build()


if __name__ == "__main__":
    print("[Quiz A2A Service] Starting on http://localhost:9001")
    print("[Quiz A2A Service] Agent Card: http://localhost:9001/.well-known/agent-card.json")
    print("[Quiz A2A Service] Press Ctrl+C to stop\n")
    uvicorn.run(create_quiz_server(), host="0.0.0.0", port=9001, log_level="warning")
