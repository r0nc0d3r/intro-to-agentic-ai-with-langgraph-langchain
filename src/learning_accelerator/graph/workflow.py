from __future__ import annotations

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from learning_accelerator.agents.curriculum_planner import curriculum_planner_node
from learning_accelerator.graph.state import AgentState

DEFAULT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", ".data/checkpoints.sqlite")


def build_graph(db_path: str = DEFAULT_DB_PATH):
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # check_same_thread=False: LangGraph runs node functions and checkpoint
    # writes on different threads, so the connection can't be thread-bound.
    # No context manager: this connection must survive the whole process,
    # not just this function call.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(AgentState)
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
