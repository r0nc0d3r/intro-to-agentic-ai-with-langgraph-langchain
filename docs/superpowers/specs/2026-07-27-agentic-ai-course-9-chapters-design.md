# Design: Intro to Agentic AI Course — 9-Chapter Build

**Date:** 2026-07-27
**Source material:** [How to Build a Multi-Agent AI System with LangGraph, MCP, and A2A](https://www.freecodecamp.org/news/how-to-build-a-multi-agent-ai-system-with-langgraph-mcp-and-a2a-full-book/) (freeCodeCamp)

## Goal

Reproduce the article's "Learning Accelerator" multi-agent system in this repo,
one chapter per iteration (9 chapters total → 9 tasks). Each chapter ends with
a flashcard-style learning summary in `learning/chapterN.md`.

## What we're building

A 4-agent LangGraph system (Curriculum Planner, Explainer, Quiz Generator,
Progress Coach) with:
- Stateful orchestration via LangGraph + SQLite checkpointing
- Tool access via MCP servers (filesystem, memory)
- Human-in-the-loop approval via `interrupt()`
- Observability via Langfuse
- Automated quality evaluation via DeepEval
- Cross-framework coordination via A2A protocol (interop with a CrewAI agent)
- A Streamlit UI tying it all together

## Repo structure (single evolving codebase)

```
src/
  config.py          # provider switch: ollama | anthropic | openai (env-driven)
  state.py            # shared LangGraph state schema
  graph.py            # graph wiring, checkpointing
  agents/
    curriculum_planner.py
    explainer.py
    quiz_generator.py
    progress_coach.py
  mcp_servers/
    filesystem_server.py
    memory_server.py
  a2a/                 # chapter 8
tests/
docs/
  architecture.md      # chapter 1 design rationale, referenced by later chapters
learning/
  chapter1.md ... chapter9.md
pyproject.toml
.env.example
```

Code accumulates in `src/` chapter over chapter — chapter N's PR builds on
chapter N-1's merged code, matching how the article itself builds the system
incrementally. No per-chapter code duplication/snapshotting.

## LLM provider: support both Ollama and hosted APIs

The article uses Ollama exclusively (no API keys needed). This repo's README
already documents OpenAI/Anthropic as a prerequisite, so we add a thin
provider-switch layer: `config.py` reads an `LLM_PROVIDER` env var
(`ollama` / `anthropic` / `openai`) and returns the matching LangChain chat
model instance. `.env.example` documents all three paths. The article's
Ollama-specific code is the default/reference path; the API path is an added
abstraction not present in the source article.

## Chapter → task → PR mapping

One task per iteration, one PR per chapter, branched as `agent/chapter-N` off
freshly-updated `main`, merged before the next chapter starts.

| Ch | Deliverable |
|----|---|
| 1 | `docs/architecture.md` (multi-agent rationale, 4-agent design case study) + `learning/chapter1.md`. No code — this chapter is architectural reasoning only. |
| 2 | Project scaffold (pyproject.toml, uv setup, `src/` package skeleton), `config.py` provider switch, shared state schema, Curriculum Planner node, graph wiring + SQLite checkpointing |
| 3 | MCP filesystem + memory servers, Explainer agent calling them in an iterative loop |
| 4 | Quiz Generator + Progress Coach agents, conditional routing that loops through all topics, end-to-end run |
| 5 | Checkpoint mechanics deep dive, human approval node (`interrupt()`), resume-after-interrupt |
| 6 | Langfuse tracing integration (LLM calls, tool invocations, node execution) |
| 7 | DeepEval LLM-as-judge quality evaluation |
| 8 | A2A protocol, cross-framework coordination with a CrewAI-built agent |
| 9 | Streamlit web UI, full system assembly, deployment notes |

Every chapter ≥2 PR also adds `learning/chapterN.md`.

## Testing approach

Light pytest coverage for pure logic only: state reducers, conditional
routing functions, config/provider selection. LLM-calling nodes get a
manual-run example script per chapter (not part of CI) since their output is
non-deterministic and requires a live Ollama instance or API key.

## Error handling

Follow the article's own patterns (retry/fallback in MCP tool calls,
checkpoint-based recovery). No extra production hardening beyond what the
article teaches — this is a learning repo, not a production system. Full
hardening is covered by the article's own Appendix C, which chapter 9 can
reference but need not fully implement.

## Learning flashcard format (`learning/chapterN.md`)

Short Q/A or term/definition pairs, technical and quick to scan — not a
full retelling of the chapter. Example:

```md
## Chapter 1: When to Use Multiple Agents

**Q: When is single-agent preferred over multi-agent?**
A: When tasks don't need separable expertise/state — single agent avoids
coordination overhead.

**Q: What's the Learning Accelerator's 4 agents?**
A: Curriculum Planner, Explainer, Quiz Generator, Progress Coach.
```

## Out of scope

- Production deployment hardening beyond article's teaching (Appendix C is
  reference material, not a chapter 9 requirement)
- Full framework comparison implementation (Appendix A/B are reading
  material, not build tasks)
- CI pipeline changes beyond running the light pytest suite
