## Chapter 2: Stateful Orchestration with LangGraph

**Q: What reducer does the `messages` field in `AgentState` use, and why?**
A: `Annotated[list[BaseMessage], add_messages]` — appends new messages
instead of overwriting the list, so conversation history accumulates
across every agent in the graph.

**Q: How do other `AgentState` fields (e.g. `roadmap`, `approved`) update?**
A: Last-write-wins — a node's returned value for that key replaces the
previous one (no reducer).

**Q: What does a node function return — the full state or a partial update?**
A: A partial dict of only the keys it changed. LangGraph merges it into
the existing state using each field's reducer (or last-write-wins).

**Q: Why is `SqliteSaver` constructed from a raw `sqlite3.Connection`
instead of `SqliteSaver.from_conn_string(...)` as a context manager?**
A: LangGraph runs node functions and checkpoint writes on different
threads (`check_same_thread=False`), and the connection must stay open
for the whole process — a `with` block would close it too early.

**Q: Why are `Topic`/`StudyRoadmap` Pydantic models here instead of the
source article's plain dataclasses?**
A: `llm.with_structured_output(StudyRoadmap)` needs a Pydantic model (or
JSON schema) to work identically across Ollama/Anthropic/OpenAI — this
repo's provider-agnostic goal the article doesn't need to solve.

**Q: What temperature does the Curriculum Planner use, and why?**
A: `0.1` — planning wants deterministic, consistent structured output,
not creative variation.

**Q: What does chapter 2's graph actually wire up?**
A: Only `START → curriculum_planner → END`. The other four agents
(explainer, quiz generator, progress coach, human approval) don't exist
yet — they're added in chapters 3-5.

**Q: What local Ollama model was used to verify this chapter, and did it
need adjusting?**
A: `gemma4:12b-mlx` failed — it ignored the structured-output constraint
and returned markdown-formatted prose instead of JSON, raising an
`OutputParserException`. Switching to `gemma4:12b` (same model family,
non-MLX build) worked cleanly and produced a valid 5-topic
`StudyRoadmap`. Lesson: not every local model quantization/build follows
tool-calling/structured-output prompting reliably — verify per-model
before assuming `with_structured_output` will work.
