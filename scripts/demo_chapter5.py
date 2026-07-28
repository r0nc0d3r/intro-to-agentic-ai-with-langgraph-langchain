"""Manual run: demonstrate interrupt / simulated-restart / resume.

Like chapter 4's demo, this monkeypatches quiz_generator._default_answer_source
with a canned response so the run completes without a real terminal attached —
there's no way to feed this session's tooling real interactive stdin. For a
genuine interactive quiz, call run_quiz() directly from a real terminal instead
(its default answer_source is real input()).

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly.
"""

from __future__ import annotations

import uuid

from langgraph.types import Command

from learning_accelerator.agents import quiz_generator
from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import build_graph, DEFAULT_DB_PATH


def _canned_answer_source(question: str) -> str:
    print(f"[canned] {question}")
    return "I'm not fully sure, but I'll give it my best guess."


def main() -> None:
    quiz_generator._default_answer_source = _canned_answer_source

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    graph = build_graph()
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)

    result = graph.invoke(state, config=config)
    assert "__interrupt__" in result, "expected the graph to stop at the interrupt"
    interrupt_payload = result["__interrupt__"][0].value
    print(f"Interrupted. Payload prompt: {interrupt_payload['prompt']!r}")
    print(f"Roadmap topics pending approval: {[t.title for t in interrupt_payload['roadmap'].topics]}")

    # Simulate a process restart: a brand new graph instance, own SqliteSaver
    # connection, same db file. If resume still works, the state genuinely
    # persisted to disk rather than living only in the first graph object.
    print("\n--- simulating process restart (fresh build_graph()) ---\n")
    restarted_graph = build_graph(db_path=DEFAULT_DB_PATH)

    result = restarted_graph.invoke(Command(resume="yes"), config=config)

    print(f"\nSession: {session_id}")
    print(f"Final topic index: {result['current_topic_index']}")
    for topic in result["roadmap"].topics:
        print(f"- {topic.title}: {topic.status}")
    print(f"Quiz results: {len(result['quiz_results'])}")


if __name__ == "__main__":
    main()
