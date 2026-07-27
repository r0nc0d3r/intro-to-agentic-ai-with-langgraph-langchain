from __future__ import annotations

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from learning_accelerator.agents.curriculum_planner import curriculum_planner_node
from learning_accelerator.agents.explainer import explainer_node
from learning_accelerator.agents.progress_coach import (
    progress_coach_node,
    route_after_coach,
)
from learning_accelerator.agents.quiz_generator import quiz_generator_node
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
    builder.add_node("explainer", explainer_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("progress_coach", progress_coach_node)

    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "explainer")
    builder.add_edge("explainer", "quiz_generator")
    builder.add_edge("quiz_generator", "progress_coach")
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {"explainer": "explainer", "end": END},
    )

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
