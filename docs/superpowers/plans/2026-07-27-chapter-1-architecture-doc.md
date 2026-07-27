# Chapter 1: When to Use Multiple Agents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/architecture.md` (the multi-agent decision framework and
4-agent design rationale for the Learning Accelerator system) and
`learning/chapter1.md` (flashcard-style notes), as the chapter 1 deliverable
of the 9-chapter agentic AI course build.

**Architecture:** This chapter is docs-only — no code, no `src/` scaffold
(deferred to chapter 2 per spec). Two content files, each self-contained,
built from original explanation (not copied from the source article — max
one attributed quote under 15 words per copyright rules).

**Tech Stack:** Markdown only.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch naming: `agent/chapter-N` (never `claude/chapter-N` — user's global CLAUDE.md requires `agent/` prefix for worktree/feature branches)
- One PR per chapter, merged into `main` before the next chapter starts (repo's `main` is protected, PR-only)
- `docs/architecture.md` is referenced by later chapters — write it as a durable reference doc, not throwaway notes
- `learning/chapterN.md` format: short Q/A or term/definition flashcards, technical, not a full retelling
- Copyright: at most one quote under 15 words from the source article, with attribution; everything else must be original phrasing/summary

---

### Task 1: Write `docs/architecture.md`

**Files:**
- Create: `docs/architecture.md`

**Interfaces:**
- Consumes: nothing (first content file in the repo)
- Produces: a reference doc later chapters (2, 4, 8) will link back to when explaining agent boundaries — must contain a stable, linkable section per agent (`## Curriculum Planner`, `## Explainer`, `## Quiz Generator`, `## Progress Coach`) so later chapters can anchor-link to them (e.g. `architecture.md#curriculum-planner`)

- [ ] **Step 1: Draft the single-vs-multi-agent decision framework section**

Write the opening section of `docs/architecture.md`:

```markdown
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
```

- [ ] **Step 2: Draft the Learning Accelerator's 4-agent case study section**

Append to `docs/architecture.md`:

```markdown
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
```

- [ ] **Step 3: Review for copyright compliance**

Read through the full `docs/architecture.md` and confirm:
- No sentence is copy-pasted from the source article
- At most one short quote (there are none in the draft above — confirm
  none were added)
- The source is attributed once, in the `## Source` section

If any sentence reads too close to the original article's phrasing,
rewrite it in different words while keeping the same technical meaning.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md
git commit -m "Add architecture doc: when to use multiple agents (chapter 1)"
```

---

### Task 2: Write `learning/chapter1.md`

**Files:**
- Create: `learning/chapter1.md`

**Interfaces:**
- Consumes: `docs/architecture.md` (Task 1) as source material to condense into flashcards
- Produces: `learning/chapter1.md`, establishing the flashcard format/header
  convention (`## Chapter N: <title>`, then `**Q: ...**` / `A: ...` pairs)
  that chapters 2-9 will follow

- [ ] **Step 1: Create the `learning/` directory and draft flashcards**

Create `learning/chapter1.md`:

```markdown
## Chapter 1: When to Use Multiple Agents

**Q: What's the default assumption — single agent or multi-agent?**
A: Single agent. Multi-agent adds coordination cost; only split when the
problem actually needs it.

**Q: When is a single agent enough?**
A: When there's one primary job that fits in one context window (e.g.
research-and-summarize, PR review, customer support, data extraction).

**Q: What are the 6 conditions that justify splitting into multiple agents?**
A: (1) distinct tool requirements, (2) divergent LLM call patterns
(single structured output vs. multi-turn tool loop), (3) different
temperature/model needs per task, (4) need for fault isolation, (5)
independent deployment needs, (6) cross-framework collaboration.

**Q: How many of those 6 conditions should typically be true before splitting?**
A: Two or more — no single condition alone usually justifies it.

**Q: What are the concrete costs of a multi-agent system?**
A: Shared state complexity (merge strategies for multiple writers),
harder debugging (failures surface after crossing agent boundaries),
latency multiplication (N agents = at least N LLM calls per run), and
more required infrastructure (checkpointing, observability, evaluation,
human oversight).

**Q: What's the gut-check heuristic for whether to split an agent boundary?**
A: If you can't explain why two tasks shouldn't be the same agent, they
probably shouldn't be split.

**Q: What are the Learning Accelerator's four agents?**
A: Curriculum Planner, Explainer, Quiz Generator, Progress Coach.

**Q: Why is the Curriculum Planner its own agent?**
A: Single deterministic call (`temperature=0.1`), structured JSON output,
no tools — mixing it with tool-calling agents would add noise.

**Q: Why is the Explainer its own agent?**
A: Multi-turn tool-calling loop (`temperature=0.3`) with a
non-deterministic iteration count — a different execution shape than a
single structured call.

**Q: Why is the Quiz Generator its own agent?**
A: Two LLM calls at different temperatures (0.4 generating, 0.1 grading),
plus it runs standalone over A2A — a dual-purpose pattern that doesn't
fit elsewhere.

**Q: Why is the Progress Coach its own agent?**
A: It makes the one cross-agent A2A call, synthesizes all other agents'
output, and owns the routing decision — coordination is its own concern.

**Q: Does this 4-agent split pattern generalize beyond learning/education?**
A: Yes — same shape (specialized agents + open protocols, only tool
access changes) appears in sales enablement, compliance training,
customer support, and engineering onboarding.
```

- [ ] **Step 2: Review flashcards against the design spec's format**

Compare against the format shown in
`docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
(short Q/A pairs, technical, not a full retelling). Trim any card that's
too long or re-explains rather than states the fact.

- [ ] **Step 3: Commit**

```bash
git add learning/chapter1.md
git commit -m "Add chapter 1 learning flashcards"
```

---

### Task 3: Open the chapter 1 PR

**Files:** none (branch/PR operation only)

**Interfaces:**
- Consumes: commits from Task 1 and Task 2
- Produces: an open PR against `main` for review/merge

- [ ] **Step 1: Confirm branch name and check current branch**

Run: `git branch --show-current`
Expected: a branch already named `agent/chapter-1` (create it before
starting Task 1 if it doesn't exist yet — branch off latest `main`,
never `claude/chapter-1`).

- [ ] **Step 2: Push the branch**

**Before running this step, ask the user to confirm** — pushing to the
remote is a visible, shared-state action.

```bash
git push -u origin agent/chapter-1
```

- [ ] **Step 3: Open the PR**

**Before running this step, ask the user to confirm.**

```bash
gh pr create --title "Chapter 1: When to Use Multiple Agents" --body "$(cat <<'EOF'
## Summary
- Add docs/architecture.md: single-vs-multi-agent decision framework and
  the Learning Accelerator's 4-agent design rationale
- Add learning/chapter1.md: flashcard-style chapter notes

## Test plan
- [ ] Read docs/architecture.md — confirm sections are clear and each
      agent's rationale is stated
- [ ] Read learning/chapter1.md — confirm flashcards are accurate and
      match the architecture doc

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report the PR URL to the user**

Share the URL returned by `gh pr create` so the user can review and
merge before chapter 2 starts.
