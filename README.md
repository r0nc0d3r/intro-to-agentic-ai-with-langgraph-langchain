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

# Install dependencies (once requirements are added)
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your provider/API keys before
running any examples (Ollama needs none by default).

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

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
