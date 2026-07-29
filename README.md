# Intro to Agentic AI with LangGraph & LangChain

A hands-on introduction to building agentic AI applications using
[LangGraph](https://github.com/langchain-ai/langgraph) and
[LangChain](https://github.com/langchain-ai/langchain).

## About

This repository collects examples, notes, and exercises for learning how to
design and build agentic workflows — from simple chains to stateful,
multi-step graphs with tool use and memory.

## Prerequisites

- Python 3.11+
- Either a local [Ollama](https://ollama.com/) install (default, no API key
  needed) or an API key for a hosted provider (Anthropic, OpenAI)

## Getting Started

Clone the repository first:

```bash
git clone https://github.com/r0nc0d3r/intro-to-agentic-ai-with-langgraph-langchain.git
cd intro-to-agentic-ai-with-langgraph-langchain
```

### Option A: uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package/project manager
that replaces `python -m venv`, `pip`, and `pip-tools` with a single tool.

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment (equivalent to python -m venv .venv)
uv venv

# Install dependencies from pyproject.toml/requirements.txt (equivalent to pip install -r requirements.txt)
uv sync

# Add a new dependency (equivalent to pip install <package> + updating requirements)
uv add <package>

# Run a script inside the project's environment without manually activating it (equivalent to source .venv/bin/activate && python script.py)
uv run python script.py

# Run a one-off CLI tool in an ephemeral environment (equivalent to pipx run <tool>)
uvx <tool>
```

### Option B: standard venv + pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the project and its dependencies from pyproject.toml
pip install -e .
```

Copy `.env.example` to `.env` and fill in your provider/API keys before
running any examples (Ollama needs none by default). Langfuse and A2A
variables are optional — see [Observability, evaluation & A2A](#observability-evaluation--a2a-chapters-6-8)
below.

## Running the System

The Learning Accelerator is a four-agent LangGraph system (Curriculum
Planner, Explainer, Quiz Generator, Progress Coach) with two real entry
points:

```bash
# Terminal interface
uv run python main.py "Learn LangGraph checkpointing from scratch"

# Resume an interrupted session by its ID
uv run python main.py --resume <session-id>

# Web interface (Streamlit)
uv run streamlit run streamlit_app.py
```

For the full build-out chapter by chapter — architecture rationale,
MCP/A2A integration, observability, evaluation, and more — see
[`docs/architecture.md`](docs/architecture.md) and the flashcard-style
notes in [`learning/`](learning/) (`chapter1.md` through `chapter9.md`).
Each chapter also has a standalone demo script under
[`scripts/`](scripts/) (e.g. `uv run python scripts/demo_chapter5.py`)
for seeing that chapter's piece run in isolation.

## Project Structure

```
src/learning_accelerator/
  agents/           Curriculum Planner, Explainer, Quiz Generator, Progress Coach, human_approval
  graph/            Shared AgentState schema + the LangGraph workflow
  mcp_servers/      Filesystem and memory MCP tool servers (chapter 3)
  a2a_services/     Quiz Generator exposed as a standalone A2A service (chapter 8)
  crewai_agent/     CrewAI "Study Buddy" — cross-framework A2A interop (chapter 8)
  evaluation/       DeepEval LLM-as-judge quality tests setup (chapter 7)
  observability/    Langfuse tracing setup (chapter 6)
docs/               Architecture rationale and implementation plans
learning/           Flashcard-style notes per chapter
scripts/            Standalone per-chapter demo scripts
study_materials/    Sample notes used by the Explainer's MCP tools
tests/              pytest suite (unit tests + opt-in eval tests)
```

## Observability, evaluation & A2A (chapters 6-8)

These are optional, off by default unless configured in `.env`:

- **Observability** — set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` to
  trace graph runs to Langfuse Cloud (or self-host via
  [`docker-compose.langfuse.yml`](docker-compose.langfuse.yml)).
- **Evaluation** — `tests/test_eval.py` runs DeepEval LLM-as-judge quality
  checks against the Explainer, Quiz Generator, and Progress Coach. These
  make real LLM calls, so they're excluded from the default test run and
  opted into explicitly (see [Running Tests](#running-tests)).
- **A2A cross-framework coordination** — the Quiz Generator can run as a
  standalone A2A service, and a CrewAI-based "Study Buddy" (a different
  framework entirely) is reachable through the same protocol. Toggle with
  `USE_A2A_QUIZ`/`USE_STUDY_BUDDY` in `.env`; see
  `scripts/demo_chapter8.py` for a real end-to-end run of both services.

## Running Tests

```bash
# Fast suite (no network calls, no LLM calls)
uv run pytest -q

# Include the LLM-as-judge evaluation tests (needs a configured provider)
uv run pytest -m eval
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
