# Intro to Agentic AI with LangGraph & LangChain

A hands-on introduction to building agentic AI applications using
[LangGraph](https://github.com/langchain-ai/langgraph) and
[LangChain](https://github.com/langchain-ai/langchain).

## About

This repository collects examples, notes, and exercises for learning how to
design and build agentic workflows — from simple chains to stateful,
multi-step graphs with tool use and memory.

## Prerequisites

- Python 3.10+
- An API key for at least one supported LLM provider (e.g. OpenAI, Anthropic)

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

Copy `.env.example` to `.env` (once added) and fill in your API keys before
running any examples.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
