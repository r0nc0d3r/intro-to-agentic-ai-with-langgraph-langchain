# State Management in LangGraph

State fields update in one of two ways:

1. **Last-write-wins** — a node's returned value for a key simply
   replaces the previous value. Most fields work this way (e.g. a
   `roadmap` or an `approved` flag).
2. **Reducer-based** — a field is annotated with a reducer function that
   controls how new values combine with old ones. The most common
   example is `messages`, annotated with `add_messages`, which appends
   new messages instead of replacing the list.

Nodes should return only the keys they actually changed — not the whole
state — so unrelated fields are left untouched by that step.
