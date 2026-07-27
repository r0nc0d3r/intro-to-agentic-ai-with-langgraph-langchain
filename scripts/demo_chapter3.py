"""Manual run: invoke the chapter 3 graph (Curriculum Planner + Explainer).

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly. The
Explainer additionally needs a model that supports tool-calling.
"""

from __future__ import annotations

import uuid

from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import graph


def main() -> None:
    session_id = str(uuid.uuid4())
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)

    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(state, config=config)

    print(f"Session: {session_id}")
    print(f"Roadmap topics: {[t.title for t in result['roadmap'].topics]}")
    print(f"Explainer output:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
