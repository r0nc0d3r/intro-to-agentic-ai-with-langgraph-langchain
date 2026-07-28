# Chapter 5: State Persistence and Human Oversight — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `human_approval` node using `interrupt()` for mid-execution
pauses, wire it between Curriculum Planner and Explainer, and demonstrate a
genuine resume-after-interrupt cycle including surviving a simulated process
restart (proving the SQLite checkpoint, not just in-process state, is what
persists).

**Architecture:** `agents/human_approval.py` adds `human_approval_node`
(calls `interrupt()` from `langgraph.types` with a payload describing the
roadmap for review) and `route_after_approval` (pure routing logic: reads
`state["approved"]`, no LLM call). `graph/workflow.py` inserts
`human_approval` between `curriculum_planner` and `explainer` via a
conditional edge — approved continues to `explainer`, rejected loops back to
`curriculum_planner` to regenerate. Resuming uses
`graph.invoke(Command(resume=value), config)`, confirmed as the current API
against the LangGraph source via context7 (`Command` dataclass in
`langgraph/types.py`, `resume` field).

**Verified via context7 (not just the source article):** on resume, "the
graph resumes from the start of the node, re-executing all logic" (the
node's Python function body reruns from its own top; `interrupt()`'s second
call inside that rerun returns the resume value instead of raising). This is
a node-local re-execution detail, not a special state-merging rule — this
codebase's established pattern of returning only changed keys from a node
(used in every prior chapter) still holds here, so `human_approval_node`
does not need to explicitly return every unrelated field the way the
article's version defensively does.

**Tech Stack:** `langgraph.types.interrupt`, `langgraph.types.Command`,
pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-5` (already created, off updated `main`)
- One PR per chapter; push, open, and merge it yourself once this chapter's
  work is verified — no need to pause for confirmation (explicitly
  authorized for chapters 5-9)
- `route_after_approval(state) -> str` is pure logic (no LLM call) → full
  pytest coverage, both branches
- `human_approval_node` itself is **not** unit-testable in isolation —
  calling `interrupt()` outside a compiled graph's execution context
  raises/misbehaves because it needs LangGraph's internal scratchpad
  config. It is verified for real via the chapter 5 demo script instead
  (same "LLM-calling/integration nodes are manual-run only" policy this
  spec already applies to every other agent node — this one is
  interrupt-calling rather than LLM-calling, but the same exclusion
  applies for the same reason: it can't run outside a real graph)
- Local dev machine has Ollama with `gemma4:12b-mlx`, `gemma4:12b`,
  `qwen3.5:2b`; chapters 2-4 all found `gemma4:12b` works — use it as the
  default verification model here too
- `learning/chapterN.md` format: short Q/A flashcards (see chapters 1-4)

---

### Task 1: Human Approval node and routing

**Files:**
- Create: `src/learning_accelerator/agents/human_approval.py`
- Test: `tests/test_human_approval.py`

**Interfaces:**
- Consumes: `AgentState` (chapter 2)
- Produces: `human_approval_node(state) -> dict`,
  `route_after_approval(state) -> str` — both wired into the graph in
  Task 2

- [ ] **Step 1: Write the failing tests**

Create `tests/test_human_approval.py`:

```python
from learning_accelerator.agents.human_approval import route_after_approval
from learning_accelerator.graph.state import initial_state


def test_route_after_approval_continues_when_approved():
    state = initial_state(goal="g", session_id="s")
    state["approved"] = True

    assert route_after_approval(state) == "explainer"


def test_route_after_approval_regenerates_when_rejected():
    state = initial_state(goal="g", session_id="s")
    state["approved"] = False

    assert route_after_approval(state) == "curriculum_planner"


def test_route_after_approval_defaults_to_regenerate_when_missing():
    state = initial_state(goal="g", session_id="s")
    del state["approved"]

    assert route_after_approval(state) == "curriculum_planner"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_human_approval.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.agents.human_approval'`

- [ ] **Step 3: Write `src/learning_accelerator/agents/human_approval.py`**

```python
from __future__ import annotations

from langgraph.types import interrupt

from learning_accelerator.graph.state import AgentState


def human_approval_node(state: AgentState) -> dict:
    roadmap = state.get("roadmap")

    decision = interrupt(
        {
            "type": "roadmap_approval",
            "roadmap": roadmap,
            "prompt": "Does this study plan look good? (yes/no)",
        }
    )

    approved = str(decision).strip().lower() in ("yes", "y", "ok", "approve")

    return {"approved": approved, "error": None}


def route_after_approval(state: AgentState) -> str:
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_human_approval.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/agents/human_approval.py tests/test_human_approval.py
git commit -m "Add human_approval node and routing"
```

---

### Task 2: Wire Human Approval into the graph

**Files:**
- Modify: `src/learning_accelerator/graph/workflow.py`

**Interfaces:**
- Consumes: `human_approval_node`/`route_after_approval` (Task 1)
- Produces: graph now runs `START → curriculum_planner → human_approval
  →(conditional)→ explainer | curriculum_planner → quiz_generator →
  progress_coach →(conditional)→ explainer | END`

- [ ] **Step 1: Edit `src/learning_accelerator/graph/workflow.py`**

Add the import:

```python
from learning_accelerator.agents.human_approval import (
    human_approval_node,
    route_after_approval,
)
```

Replace:

```python
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "explainer")
```

with:

```python
    builder.add_node("human_approval", human_approval_node)

    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "human_approval")
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"explainer": "explainer", "curriculum_planner": "curriculum_planner"},
    )
```

(Add the `builder.add_node("human_approval", human_approval_node)` line
immediately after the existing `builder.add_node(...)` calls, before the
edges — match the existing node-registration block's style.)

- [ ] **Step 2: Verify the graph still compiles**

Run: `rm -f .data/checkpoints.sqlite && uv run python -c "from learning_accelerator.graph.workflow import graph; print(graph)"`
Expected: prints a `CompiledStateGraph` object, no errors

- [ ] **Step 3: Run the full pytest suite**

Run: `uv run pytest -v`
Expected: all 34 tests pass (31 existing + 3 new from Task 1 — this task
only changes graph wiring)

- [ ] **Step 4: Commit**

```bash
git add src/learning_accelerator/graph/workflow.py
git commit -m "Wire human_approval into graph between Curriculum Planner and Explainer"
```

---

### Task 3: Manual demo script + real verification (interrupt, restart, resume)

**Files:**
- Create: `scripts/demo_chapter5.py`

**Interfaces:**
- Consumes: `graph`, `build_graph`, `initial_state` (chapter 2/Task 2),
  `quiz_generator._default_answer_source` (chapter 4, monkeypatched here)

This step demonstrates three things the article calls out as this
chapter's actual teaching point, not just "the code runs":
1. The first `graph.invoke()` call stops at the interrupt and returns a
   result containing `"__interrupt__"`.
2. **Simulated crash recovery**: a brand new `build_graph()` call (a fresh
   `CompiledStateGraph`, fresh `SqliteSaver`/`sqlite3.connect`, same
   `db_path`) can still resume the same `thread_id` — proving the SQLite
   file persisted the state, not just Python-process memory.
3. `Command(resume="yes")` on that fresh graph instance continues
   execution through the rest of the four-agent loop to completion.

- [ ] **Step 1: Write `scripts/demo_chapter5.py`**

```python
"""Manual run: demonstrate interrupt / simulated-restart / resume.

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
```

- [ ] **Step 2: Run it for real against local Ollama**

This runs the full topic loop after resuming, so expect several minutes
(run in the background rather than blocking, same as chapter 4).

```bash
rm -f .data/checkpoints.sqlite
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b uv run python scripts/demo_chapter5.py
```

Expected: prints the interrupt payload and roadmap topics, then the
restart marker, then completes the full loop (every topic reaching
`completed` or `needs_review`, matching chapter 4's demo behavior). If
`gemma4:12b` doesn't cooperate, try `qwen3.5:2b` and note which model
worked in the chapter 5 flashcards, same as prior chapters.

- [ ] **Step 3: Run the full pytest suite once more**

Run: `uv run pytest -v`
Expected: all tests still pass (this step only adds a script).

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_chapter5.py
git commit -m "Add chapter 5 manual demo script (interrupt, restart, resume)"
```

---

### Task 4: Chapter 5 learning flashcards

**Files:**
- Create: `learning/chapter5.md`

- [ ] **Step 1: Write `learning/chapter5.md`**

```markdown
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

**Q: What actually happens inside the node on resume — does execution
continue from the `interrupt()` line, or does something else happen?**
A: The node's function body re-executes from the top. `interrupt()`'s
call re-fires, but this time (since a resume value is recorded) it
returns that value instead of raising — so any code before the
`interrupt()` call in the node runs again too.

**Q: Given that the node re-executes from the top, why doesn't
`human_approval_node` need to explicitly return every field like the
source article's version does?**
A: Re-execution is a node-local detail (the function body reruns) — it
isn't a special state-merging rule. Every node in this codebase already
returns only the keys it changed, and LangGraph's normal partial-update
merging (proven since chapter 2) applies here exactly the same way.

**Q: What does `route_after_approval` do?**
A: Pure logic, no LLM call: `"explainer"` if `state["approved"]` is
true, else `"curriculum_planner"` (regenerate the roadmap).

**Q: What does the chapter 5 demo prove that a normal interrupt/resume
test wouldn't?**
A: That a *brand new* `build_graph()` call — a fresh compiled graph, a
fresh `SqliteSaver`, a fresh `sqlite3.connect()` — can resume the same
`thread_id` and pick up right where the interrupt left off. That's
genuine crash recovery: the state persisted to the SQLite file, not just
to the first Python process's memory.

**Q: What does chapter 5's graph look like now?**
A: `START → curriculum_planner → human_approval →(conditional)→
explainer | curriculum_planner`, then unchanged from chapter 4:
`explainer → quiz_generator → progress_coach →(conditional)→ explainer |
END`.

**Q: Which local Ollama model was used to verify the interrupt/restart/resume
cycle?**
A: See the demo script run output for this repo.
```

Before committing, fill in the last flashcard's answer with what Task 3
actually observed.

- [ ] **Step 2: Commit**

```bash
git add learning/chapter5.md
git commit -m "Add chapter 5 learning flashcards"
```

---

### Task 5: Push, open, and merge the chapter 5 PR

**Files:** none (branch/PR operation only)

Full autonomy is authorized for chapters 5-9 — no confirmation needed
before push/PR/merge.

- [ ] **Step 1: Confirm all tests pass**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin agent/chapter-5
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "Chapter 5: State Persistence and Human Oversight" --body "$(cat <<'EOF'
## Summary
- Add human_approval node using interrupt() from langgraph.types, plus
  route_after_approval (pure logic, full pytest coverage)
- Wire human_approval between Curriculum Planner and Explainer: approved
  continues, rejected loops back to regenerate the roadmap
- Add chapter 5 demo script proving genuine crash recovery: a brand new
  build_graph() call (fresh SqliteSaver, fresh sqlite3 connection) resumes
  the same thread_id after Command(resume=...), not just in-process state
- Add learning/chapter5.md flashcards, including why this codebase's
  return-only-changed-keys convention holds even across the interrupt
  boundary (verified against the LangGraph source via context7)

## Test plan
- [ ] `uv sync && uv run pytest -v` — all tests pass
- [ ] `uv run python scripts/demo_chapter5.py` — stops at the interrupt,
      simulates a restart, resumes, and completes the full topic loop

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Merge**

```bash
gh pr merge --merge
```

- [ ] **Step 5: Delete the remote branch and sync the local worktree**

```bash
git push origin --delete agent/chapter-5
git checkout claude/agentic-ai-langgraph-course-ebf06b
git fetch origin
git merge --ff-only origin/main
git branch -d agent/chapter-5
```
