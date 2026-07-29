# Runtime Performance Pass

## Goal

Improve runtime performance of the Learning Accelerator for its primary
use case — a single user running a study session locally (terminal or
Streamlit) against Ollama. Not in scope: install/CI/dependency-footprint
optimization, or concurrent multi-user deployment concerns (both were
explicitly deprioritized during brainstorming in favor of single-session
latency).

Implemented as three small, independent chunks — each safe to ship on
its own, in its own PR, in the order below. No chunk depends on a later
one.

## Chunk 1 — Cache LLM client construction

**Problem:** `get_chat_model()` in `src/learning_accelerator/config.py`
constructs a brand-new `ChatOllama`/`ChatAnthropic`/`ChatOpenAI` instance
on every call. It's called at least once per graph node per turn — six
call sites total: `curriculum_planner_node`, `explainer_node`,
`quiz_generator.generate_questions`, `quiz_generator.grade_answer`,
`progress_coach.get_coaching_message`, and
`evaluation/judge_model.py`'s DeepEval judge wrapper. Each construction
does real work (client/session setup) that's pure overhead when the
same `(provider, temperature)` combination repeats within a session —
which it does, since there are only ~4 distinct temperatures used
across the whole app (0.0, 0.1, 0.3, 0.4).

**Change:** decorate `get_chat_model` with `functools.lru_cache`:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    ...  # body unchanged
```

The cache key is effectively `(temperature,)` — provider, model name,
and base URL are all read from environment variables that are fixed for
the lifetime of the process (`load_dotenv()` runs once at import), so
they don't need to be part of the key.

No call sites change. `.with_structured_output(...)` / `.bind_tools(...)`
are still chained fresh on every call (cheap, local object construction,
no I/O) — only the underlying client construction is memoized.

**Tests:** add a test asserting `get_chat_model(0.1) is get_chat_model(0.1)`
(cache hit) and that a different temperature returns a distinct instance.
Existing tests that `monkeypatch` `get_chat_model` per-module are
unaffected (they patch the imported name, not the cache internals).

**Risk:** near zero. LangChain chat model instances are stateless across
`.invoke()` calls; sharing one across call sites within a process is
safe.

## Chunk 2 — Extract duplicated "last AIMessage" scan

**Problem:** the same loop —

```python
for msg in reversed(state["messages"]):
    if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
        explanation = msg.content
        break
```

— appears verbatim in both `quiz_generator.py::quiz_generator_node` and
`progress_coach.py::progress_coach_node`.

**Change:** add a shared helper to `src/learning_accelerator/graph/state.py`,
next to the existing `get_current_topic()`/`session_is_complete()`
helpers (same pattern, same file):

```python
def get_last_explanation(state: AgentState) -> str:
    """Return the most recent non-tool-call AIMessage's content, or ""."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""
```

Both call sites replace their inline loop with `get_last_explanation(state)`.

**Tests:** one new unit test for the helper (empty messages, only
tool-call messages present, normal case with a real explanation). The
two call sites' existing tests are unaffected — behavior is identical,
just relocated.

**Risk:** none — pure extraction, no logic change.

## Chunk 3 — Lazy graph construction

**Problem:** `src/learning_accelerator/graph/workflow.py` ends with:

```python
graph = build_graph()
```

This runs at *import* time, meaning merely importing the module opens a
SQLite connection to `.data/checkpoints.sqlite` (creating the file/dir
if needed) — even if the caller never actually invokes the graph.

**Change:** replace the eager singleton with a lazy, cached accessor:

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_default_graph():
    return build_graph()
```

**Call sites to update** (confirmed by grep — these are the only
consumers of the old module-level `graph`):

- `main.py`
- `scripts/demo_chapter2.py`
- `scripts/demo_chapter3.py`
- `scripts/demo_chapter4.py`

Each changes `from learning_accelerator.graph.workflow import graph` to
`from learning_accelerator.graph.workflow import get_default_graph`, and
`graph.invoke(...)` to `get_default_graph().invoke(...)`.

`streamlit_app.py` and `tests/test_workflow.py` already call
`build_graph()` directly with their own arguments (a second instance
compiled with `interrupt_before` for the UI, and fresh temp-dir
instances for tests, respectively) — both unaffected by this change.

**Tests:** existing workflow tests unaffected. Add a test confirming
that importing `learning_accelerator.graph.workflow` alone does not
create `.data/checkpoints.sqlite` — only calling `get_default_graph()`
does.

**Risk:** low — mechanical rename at 4 call sites; behavior identical
once the graph is actually invoked.

## Explicitly out of scope (considered, not included)

- **A2A availability-probe latency** (`is_study_buddy_available` /
  `is_quiz_service_available` in `progress_coach.py`): investigated
  during brainstorming — `discover_agent()` hits `localhost` with a 5s
  timeout, but when the service isn't running this fails via immediate
  connection-refused, not a real timeout stall. Not a meaningful
  single-session latency issue in practice; not worth the churn.
- **`state["messages"]` unbounded growth** across a session: real, but
  at the current scale (4-6 topics per roadmap, a handful of messages
  each) it's not a measurable latency issue for a single local session.
  Revisit if roadmaps grow much larger or sessions get reused across
  many goals.
- **Dependency footprint / install speed** (splitting chapter 6-8's
  langfuse/deepeval/crewai/a2a-sdk into `[project.optional-dependencies]`
  extras): a real and separate opportunity, but out of scope — this pass
  is scoped to runtime performance, not install/CI speed, per an
  explicit choice made during brainstorming.
- **In-memory MCP `_store` in `memory_server.py` has no eviction/TTL**:
  a real concern for a long-running multi-user deployment, but out of
  scope since this pass targets single-user local sessions, where the
  process (and thus the store) has a naturally short lifetime.
