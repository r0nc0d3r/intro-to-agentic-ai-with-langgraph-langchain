## Chapter 5: State Persistence and Human Oversight

**Q: What does `interrupt()` do when called inside a node?**
A: On its first call within a run, it raises a `GraphInterrupt` that halts
execution and surfaces the given payload to the caller — the
`graph.invoke()` result contains an `"__interrupt__"` key with that
payload instead of completing normally.

**Q: How does a caller resume execution after an interrupt?**
A: `graph.invoke(Command(resume=value), config)` with the same
`thread_id` in `config` — `value` becomes what `interrupt()` returns
inside the node on the next run.

**Q: Given that the node re-executes from the top on resume, why doesn't
`human_approval_node` need to explicitly return every field like the
source article's version does?**
A: Re-execution is a node-local detail (the function body reruns) — it
isn't a special state-merging rule. Every node in this codebase already
returns only the keys it changed, and LangGraph's normal partial-update
merging (proven since chapter 2) applies here exactly the same way.

**Q: What does `route_after_approval` do?**
A: Pure logic, no LLM call: `"explainer"` if `state["approved"]` is
true, else `"curriculum_planner"` (regenerate the roadmap).

**Q: What did the chapter 5 demo prove that a normal interrupt/resume
test wouldn't?**
A: That a *brand new* `build_graph()` call — a fresh compiled graph, a
fresh `SqliteSaver`, a fresh `sqlite3.connect()` — resumed the same
`thread_id` after `Command(resume="yes")` and completed the whole
5-topic loop. That's genuine crash recovery: the state persisted to the
SQLite file, not just to the first Python process's memory.

**Q: What real, previously-unknown issue did running this for real
surface?**
A: LangGraph logs `Deserializing unregistered type
learning_accelerator.graph.state.StudyRoadmap from checkpoint. This
will be blocked in a future version.` — because `StudyRoadmap` is a
Pydantic model stored directly in checkpointed state, not a plain
dict/allow-listed type. It works today (only a deprecation warning), but
a future LangGraph version will block it outright unless the type is
registered via `allowed_msgpack_modules`. This is a chapter 2 design
choice (storing Pydantic instances in `AgentState`) that only became
visible once we actually exercised a real disk round-trip — flagged as a
tracked follow-up, not fixed inline here.

**Q: What does chapter 5's graph look like now?**
A: `START → curriculum_planner → human_approval →(conditional)→
explainer | curriculum_planner`, then unchanged from chapter 4:
`explainer → quiz_generator → progress_coach →(conditional)→ explainer |
END`.

**Q: Which local Ollama model was used to verify the interrupt/restart/resume
cycle?**
A: `gemma4:12b` — worked on the first successful attempt, no fallback
needed. (One run attempt died silently with no output, traced to a
buffering/process-lifecycle issue from how the command was launched, not
a code bug — resolved by rerunning fully detached with unbuffered stdout.)
