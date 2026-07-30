"""
main.py

Terminal entry point for the Learning Accelerator.

Usage:
  uv run python main.py "Learn LangGraph checkpointing from scratch"
  uv run python main.py --resume <session-id>
"""

from __future__ import annotations

import argparse
import uuid

from langgraph.types import Command

from learning_accelerator.graph.state import QuizResult, StudyRoadmap, initial_state
from learning_accelerator.graph.workflow import get_default_graph
from learning_accelerator.observability.langfuse_setup import flush_langfuse, get_langfuse_config


def print_session_summary(result: dict) -> None:
    """Print a summary of a completed session. No-op if there's no roadmap."""
    roadmap: StudyRoadmap | None = result.get("roadmap")
    if roadmap is None:
        return

    quiz_results: list[QuizResult] = result.get("quiz_results", [])
    if not quiz_results:
        return

    print(f"\n{'=' * 60}")
    print("Session Summary")
    print(f"{'=' * 60}")
    print(f"Goal: {roadmap.goal}")
    print(f"Topics covered: {len(quiz_results)}/{len(roadmap.topics)}")

    avg = sum(r.score for r in quiz_results) / len(quiz_results)
    print(f"Average score: {avg:.0%}\n")

    for r in quiz_results:
        status = "✓" if r.score >= 0.5 else "✗"
        weak = f", review: {', '.join(r.weak_areas)}" if r.weak_areas else ""
        print(f"  {status} {r.topic}: {r.score:.0%}{weak}")

    all_weak = result.get("weak_areas", [])
    if all_weak:
        print(f"\nTopics to revisit: {', '.join(all_weak)}")

    print(f"{'=' * 60}\n")


def run_session(goal: str, session_id: str | None = None) -> None:
    """Run a complete interactive study session with Langfuse tracing."""
    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    config = get_langfuse_config(session_id)

    print(f"\n{'=' * 60}")
    print("Learning Accelerator")
    print(f"Session ID: {session_id}")
    print("Resuming existing session..." if is_resume else f"Goal: {goal}")
    print(f"{'=' * 60}")

    state = None if is_resume else initial_state(goal, session_id)

    try:
        result = get_default_graph().invoke(state, config=config)
    except Exception as e:
        if is_resume:
            print(f"\n[ERROR] Could not resume session '{session_id}': {e}")
            print("If the session ID is wrong or the checkpoint database has "
                  "been deleted, start a new session instead.")
            return
        raise

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        roadmap: StudyRoadmap | None = interrupt_payload.get("roadmap")

        if roadmap:
            print(f"\n{'=' * 60}")
            print("Proposed Study Plan")
            print(f"{'=' * 60}")
            print(f"Goal: {roadmap.goal}")
            print(f"Duration: {roadmap.total_weeks} weeks @ {roadmap.weekly_hours} hrs/week\n")
            for i, topic in enumerate(roadmap.topics, 1):
                prereqs = f" (needs: {', '.join(topic.prerequisites)})" if topic.prerequisites else ""
                print(f"  {i}. {topic.title} ({topic.estimated_minutes} min){prereqs}")
                print(f"     {topic.description}")

        print(f"\n{interrupt_payload.get('prompt', 'Continue?')}")
        user_input = input("> ").strip()

        result = get_default_graph().invoke(Command(resume=user_input), config=config)

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        return

    print_session_summary(result)
    flush_langfuse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Learning Accelerator: a four-agent study system that plans a "
            "curriculum, explains topics from your notes, quizzes you, and "
            "adapts based on results. All inference runs locally via Ollama "
            "by default."
        ),
        epilog=(
            "Examples:\n"
            '  uv run python main.py "Learn LangGraph checkpointing from scratch"\n'
            "  uv run python main.py --resume a3f1b2c4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="Learn the basics of LangGraph",
        help="What you want to learn (default: a LangGraph starter goal)",
    )
    parser.add_argument(
        "--resume", metavar="SESSION_ID", help="Resume an existing session by its ID"
    )
    args = parser.parse_args()

    if args.resume:
        run_session(goal="", session_id=args.resume)
    else:
        run_session(goal=args.goal)
