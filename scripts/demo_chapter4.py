"""Manual run: invoke the full chapter 4 graph (all four agents), looping
through every roadmap topic to completion.

This demo monkeypatches quiz_generator._default_answer_source with a
canned response so the run completes without a real terminal attached —
there's no way to feed this session's tooling real interactive stdin. For
a genuine interactive quiz, call run_quiz() directly from a real terminal
instead (its default answer_source is real input()).

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly.
"""

from __future__ import annotations

import uuid

from learning_accelerator.agents import quiz_generator
from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import get_default_graph


def _canned_answer_source(question: str) -> str:
    print(f"[canned] {question}")
    return "I'm not fully sure, but I'll give it my best guess."


def main() -> None:
    quiz_generator._default_answer_source = _canned_answer_source

    session_id = str(uuid.uuid4())
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    result = get_default_graph().invoke(state, config=config)

    print(f"Session: {session_id}")
    print(f"Final topic index: {result['current_topic_index']}")
    for topic in result["roadmap"].topics:
        print(f"- {topic.title}: {topic.status}")
    print(f"Quiz results: {len(result['quiz_results'])}")
    for qr in result["quiz_results"]:
        print(f"  {qr.topic}: score={qr.score:.2f} passed={qr.passed}")


if __name__ == "__main__":
    main()
