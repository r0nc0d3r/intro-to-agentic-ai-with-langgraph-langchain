# Architecture: When to Use Multiple Agents

## The core question

Before splitting a system into multiple agents, ask: does this problem
actually need more than one? Multi-agent is not automatically better —
it adds coordination cost that a single agent avoids entirely.

A single agent is usually enough when there's one primary job that fits
in one context window: research-and-summarize, PR review, customer
support, data extraction. These fit inside one LLM call or one
iterative tool-calling loop without needing separation.

## When to split into multiple agents

Split when two or more of the following are true for your problem:

1. **Distinct tool requirements** — subtasks need fundamentally different
   tool access (filesystem vs. database vs. external API), creating a
   natural boundary.
2. **Divergent call patterns** — one subtask needs one structured output,
   another needs a multi-turn tool-calling loop. Bundling them into one
   function makes it fail differently depending on which path executes.
3. **Temperature/model variance** — planning wants deterministic output
   (`temperature=0`), creative generation wants variation
   (`temperature≈0.3-0.4`), grading wants precision (`temperature≈0.1`).
   One shared setting forces a compromise on all of them.
4. **Fault isolation** — one subtask failing shouldn't stop the others.
   Agent boundaries contain the blast radius.
5. **Independent deployment** — different scale needs, update cadence, or
   ownership map naturally onto separate agents/services.
6. **Cross-framework collaboration** — agents built in different
   frameworks (e.g. LangGraph and CrewAI) need a protocol boundary
   between them regardless of anything else.

No single condition mandates separation on its own — but two or more
usually do.

## The cost of splitting

Multi-agent isn't free. Every agent boundary adds:

- **Shared state complexity** — multiple writers to shared state need a
  merge strategy; state shape becomes a contract every agent depends on.
- **Harder debugging** — a failure can surface several steps after its
  actual cause, once it's crossed an agent boundary.
- **Latency multiplication** — N agents means at least N LLM calls per
  run; at a few seconds each, this adds up fast.
- **More infrastructure** — checkpointing, observability, evaluation, and
  human-oversight hooks that a single agent could often skip become
  necessary once there are multiple moving parts to coordinate.

A useful gut check before adding an agent boundary: if you can't explain
*why* two tasks shouldn't be the same agent, they probably shouldn't be
split.

## Case study: the Learning Accelerator's four agents

The Learning Accelerator splits into four agents because each one has a
distinct execution pattern — not because "four agents sounds thorough."

### Curriculum Planner

- One deterministic LLM call, `temperature=0.1`
- Produces structured JSON (a topic list/plan), no tool use
- Fast, few failure modes
- *Why separate:* mixing a structured-JSON-only task with tool-calling
  agents would add output-formatting noise neither needs.

### Explainer

- Multi-turn tool-calling loop, `temperature=0.3`
- Iteration count is non-deterministic — the LLM decides when it's done
- Reads study materials through MCP tool calls
- *Why separate:* a completely different execution shape than the
  Planner's single call; needs its own orchestration loop.

### Quiz Generator

- Two LLM calls at different temperatures: `temperature=0.4` for
  generating questions (creative), `temperature=0.1` for grading answers
  (analytical)
- Pauses for interactive input/output
- Runs as a standalone service reachable over A2A
- *Why separate:* the dual-temperature, dual-purpose pattern plus
  standalone deployment needs don't fit inside another agent.

### Progress Coach

- Makes the system's one cross-agent A2A call
- Synthesizes every other agent's output
- Owns the routing decision that controls graph flow
- *Why separate:* coordination responsibility is its own concern — it
  shouldn't be bundled with any task it's coordinating.

### The pattern generalizes

The same shape — specialized agents coordinating through open protocols,
only the tool-access points changing — shows up in sales enablement
(onboarding → training adaptation), compliance training (curriculum →
delivery → assessment), customer support (knowledge base → escalation),
and engineering onboarding (codebase walkthroughs). The four-agent split
here is a specific instance of a general pattern, not a one-off.

## Source

Design reasoning adapted from ["How to Build a Multi-Agent AI System with
LangGraph, MCP, and A2A"](https://www.freecodecamp.org/news/how-to-build-a-multi-agent-ai-system-with-langgraph-mcp-and-a2a-full-book/)
(freeCodeCamp), Chapter 1.
