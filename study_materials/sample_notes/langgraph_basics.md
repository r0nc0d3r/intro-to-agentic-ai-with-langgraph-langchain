# LangGraph Basics

LangGraph models an agent as a graph of nodes and edges over a shared
state object. Each node is a plain function that reads the current state
and returns a partial update — LangGraph merges that update back in.

Key ideas:

- **Nodes** are functions: `(state) -> dict`.
- **Edges** connect nodes and can be conditional (a routing function
  decides which node runs next based on the current state).
- **State** is typically a `TypedDict` describing every field any node
  might read or write.
- **Checkpointing** persists state after each step, so a run can be
  paused, resumed, or recovered after a crash.

A graph always has a `START` and at least one path to `END`.
