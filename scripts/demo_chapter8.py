"""Manual run: start both A2A services for real, then call each one over
real HTTP — proving genuine cross-framework coordination (a LangGraph
service and a CrewAI service, both behind the same A2A protocol).

Requires local Ollama running (OLLAMA_MODEL, default gemma4:12b) — used
by both the Quiz Generator's LangChain calls and the CrewAI Study Buddy's
LLM calls.
"""

from __future__ import annotations

import multiprocessing
import time

import uvicorn

from learning_accelerator.a2a_services.a2a_client import (
    delegate_quiz_task,
    discover_agent,
    request_study_assistance,
)


def _run_quiz_server() -> None:
    from learning_accelerator.a2a_services.quiz_service import create_quiz_server

    uvicorn.run(create_quiz_server(), host="127.0.0.1", port=9001, log_level="warning")


def _run_study_buddy_server() -> None:
    from learning_accelerator.crewai_agent.study_buddy import create_study_buddy_server

    uvicorn.run(create_study_buddy_server(), host="127.0.0.1", port=9002, log_level="warning")


def _wait_until_healthy(url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if discover_agent(url):
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    quiz_process = multiprocessing.Process(target=_run_quiz_server, daemon=True)
    study_buddy_process = multiprocessing.Process(target=_run_study_buddy_server, daemon=True)

    quiz_process.start()
    study_buddy_process.start()

    try:
        print("Waiting for both A2A services to come up...")
        assert _wait_until_healthy("http://localhost:9001"), "Quiz service never became healthy"
        assert _wait_until_healthy("http://localhost:9002"), "Study Buddy service never became healthy"

        quiz_card = discover_agent("http://localhost:9001")
        study_buddy_card = discover_agent("http://localhost:9002")
        print(f"Quiz service card: {quiz_card['name']} (skills: {[s['id'] for s in quiz_card['skills']]})")
        print(f"Study Buddy card: {study_buddy_card['name']} (skills: {[s['id'] for s in study_buddy_card['skills']]})")

        print("\n--- Delegating quiz generation to the Quiz A2A service ---")
        quiz_result = delegate_quiz_task(
            topic="LangGraph Checkpointing",
            explanation=(
                "A checkpoint is a saved snapshot of the graph's state after a "
                "step, persisted so a run can be paused, resumed, or recovered "
                "after a crash."
            ),
        )
        print(f"Quiz result status: {quiz_result.get('status')}")
        assert quiz_result.get("status") == "questions_ready", f"Unexpected quiz result: {quiz_result}"
        print(f"Questions generated: {len(quiz_result.get('questions', []))}")

        print("\n--- Requesting supplementary help from the CrewAI Study Buddy (real crew.kickoff()) ---")
        assistance = request_study_assistance(
            topic="LangGraph Checkpointing",
            explanation=(
                "A checkpoint is a saved snapshot of the graph's state after a "
                "step, persisted so a run can be paused, resumed, or recovered "
                "after a crash."
            ),
            weak_areas=["thread_id", "SqliteSaver"],
        )
        print(f"Study Buddy status: {assistance.get('status')}")
        assert assistance.get("status") == "complete", f"Unexpected Study Buddy result: {assistance}"
        print(f"Assistance text ({len(assistance.get('assistance', ''))} chars):")
        print(assistance.get("assistance", ""))

    finally:
        quiz_process.terminate()
        study_buddy_process.terminate()
        quiz_process.join(timeout=5)
        study_buddy_process.join(timeout=5)


if __name__ == "__main__":
    main()
