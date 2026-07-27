from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from learning_accelerator.config import get_chat_model
from learning_accelerator.graph.state import AgentState
from learning_accelerator.mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
)
from learning_accelerator.mcp_servers.memory_server import memory_get, memory_set

MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are the Explainer. Use the available tools to read \
study materials and explain the current topic clearly to the learner. \
Save a short summary to memory under the key "last_explanation" when \
you're done. Stop calling tools once you've given your final explanation."""


@tool
def tool_list_files() -> list[str]:
    """List available study note filenames."""
    return list_study_files()


@tool
def tool_read_file(filename: str) -> str:
    """Read a study note by filename."""
    return read_study_file(filename)


@tool
def tool_search_notes(query: str) -> list[str]:
    """Search study notes for a substring, case-insensitive."""
    return search_notes(query)


@tool
def tool_memory_get(session_id: str, key: str) -> str:
    """Retrieve a value from session memory."""
    return memory_get(session_id, key)


@tool
def tool_memory_set(session_id: str, key: str, value: str) -> str:
    """Store a value in session memory."""
    return memory_set(session_id, key, value)


EXPLAINER_TOOLS = [
    tool_list_files,
    tool_read_file,
    tool_search_notes,
    tool_memory_get,
    tool_memory_set,
]

_TOOLS_BY_NAME = {t.name: t for t in EXPLAINER_TOOLS}


def _execute_tool_call(tool_call: dict) -> str:
    tool_fn = _TOOLS_BY_NAME[tool_call["name"]]
    result = tool_fn.invoke(tool_call["args"])
    return str(result)


def explainer_node(state: AgentState) -> dict:
    llm = get_chat_model(temperature=0.3).bind_tools(EXPLAINER_TOOLS)

    topic = state["roadmap"].topics[state["current_topic_index"]]
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                f"Session ID: {state['session_id']}. "
                f"Current topic: {topic.title} — {topic.description}"
            )
        ),
    ]

    final_response = None
    for _ in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            final_response = response
            break

        for tool_call in response.tool_calls:
            result = _execute_tool_call(tool_call)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    if final_response is None:
        return {"error": "explainer exceeded max iterations"}

    return {"messages": [final_response], "error": None}
