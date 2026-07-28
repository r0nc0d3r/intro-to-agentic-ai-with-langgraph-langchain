# Chapter 6: Observability with Langfuse — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Langfuse's LangChain callback handler into graph runs so every
LLM call, tool call, and node execution gets traced, using the same
`config`-dict pattern the article uses (`get_langfuse_config(session_id)`),
with graceful no-op behavior when Langfuse credentials aren't configured.

**Architecture:** `observability/langfuse_setup.py` provides
`langfuse_enabled()`, `get_langfuse_handler()`, and
`get_langfuse_config(session_id)`. The handler is `CallbackHandler` from
`langfuse.langchain` (confirmed current/non-deprecated import path via
context7 — not the older `langfuse.callback` path). Session linkage uses
the `langfuse_session_id` metadata key, which `CallbackHandler` reads and
maps to Langfuse's `session_id` trace attribute (confirmed against the
Langfuse Python SDK source) — this ties a Langfuse trace to our
`thread_id`.

**Verified via context7 (important, non-obvious mechanics):** LangGraph
propagates the invoke-time `config` (including `callbacks`) into every
node's execution context via Python contextvars
(`var_child_runnable_config`), and LangChain's `ChatModel.invoke()` reads
that ambient config through `ensure_config()` when no explicit `config` is
passed. This means every existing agent node in this codebase — none of
which currently accept or forward a `config` parameter — will still have
its LLM calls traced automatically once `callbacks` is set at
`graph.invoke()` time. No changes to `curriculum_planner.py`,
`explainer.py`, `quiz_generator.py`, or `progress_coach.py` are needed for
tracing to reach their LLM calls.

**Deliberate scope decision (read before starting):** A real self-hosted
Langfuse instance needs the full v3 stack — postgres + clickhouse + redis
+ minio + langfuse-worker + langfuse-web — which is too heavy to stand up
just to verify a callback wiring (multi-GB image pulls, slow health
checks, and it would leave a large persistent footprint on the dev
machine for a one-time check). Instead:
1. `docker-compose.langfuse.yml` is added as a **reference file** a user
   can run themselves if they want a real local Langfuse instance — it is
   not run as part of this chapter's verification.
2. Real verification instead spins up a tiny local HTTP server that
   stands in for Langfuse's ingestion endpoint, points `LANGFUSE_HOST` at
   it, runs one real graph call through `get_langfuse_config()`, and
   confirms the mock server actually received a POST request — this
   proves our integration code genuinely fires network calls at the
   configured host (the actual thing our code is responsible for), without
   needing to verify Langfuse's own server-side trace rendering (which is
   Langfuse's code, not ours).
3. The demo only needs to run through `curriculum_planner` (one real LLM
   call) before hitting the existing `human_approval` interrupt — that's
   enough to prove a traced LLM call fired, so this demo does not need the
   canned-answer monkeypatch or a multi-minute full loop like chapters 4-5.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-6` (already created, off updated `main`)
- One PR per chapter; push, open, and merge it yourself once verified —
  full autonomy already authorized for chapters 5-9
- `langfuse_enabled()`, `get_langfuse_handler()` returning `None` when
  disabled, and `get_langfuse_config()`'s dict shape are all pure logic
  (constructing a `CallbackHandler()` does not make a network call at
  construction time — it's a lazy/batched client) → full pytest coverage
- Real network-call verification (the mock ingestion server) is a
  manual-run demo script, not a pytest test, per this spec's established
  "LLM-calling/integration is manual-run only" policy — a live HTTP
  server + real Ollama call is integration-level, not pure logic
- `.env.example` already has `LLM_PROVIDER`, `OLLAMA_*`, `ANTHROPIC_*`,
  `OPENAI_*`, `CHECKPOINT_DB_PATH` — add `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` alongside them, all empty by
  default (matches the article's own `.env.example` pattern)
- `learning/chapterN.md` format: short Q/A flashcards (see chapters 1-5)

---

### Task 1: Langfuse setup module + tests

**Files:**
- Create: `src/learning_accelerator/observability/__init__.py`
- Create: `src/learning_accelerator/observability/langfuse_setup.py`
- Test: `tests/test_langfuse_setup.py`

**Interfaces:**
- Consumes: nothing
- Produces: `langfuse_enabled() -> bool`,
  `get_langfuse_handler() -> CallbackHandler | None`,
  `get_langfuse_config(session_id: str) -> dict` — used by the chapter 6
  demo script in Task 3, and available for any later chapter's scripts to
  adopt in place of a bare `{"configurable": {"thread_id": ...}}` dict

- [ ] **Step 1: Add the langfuse dependency**

```bash
uv add langfuse
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_langfuse_setup.py`:

```python
from learning_accelerator.observability.langfuse_setup import (
    get_langfuse_config,
    get_langfuse_handler,
    langfuse_enabled,
)


def test_langfuse_enabled_false_when_keys_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert langfuse_enabled() is False


def test_langfuse_enabled_false_when_only_public_key_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert langfuse_enabled() is False


def test_langfuse_enabled_true_when_both_keys_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    assert langfuse_enabled() is True


def test_get_langfuse_handler_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert get_langfuse_handler() is None


def test_get_langfuse_handler_returns_handler_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:59999")

    handler = get_langfuse_handler()
    assert handler is not None


def test_get_langfuse_config_omits_callbacks_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    config = get_langfuse_config("session-123")

    assert config == {"configurable": {"thread_id": "session-123"}}
    assert "callbacks" not in config


def test_get_langfuse_config_includes_callbacks_when_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:59999")

    config = get_langfuse_config("session-456")

    assert config["configurable"]["thread_id"] == "session-456"
    assert len(config["callbacks"]) == 1
    assert config["metadata"] == {"langfuse_session_id": "session-456"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_langfuse_setup.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.observability'`

- [ ] **Step 4: Create `src/learning_accelerator/observability/__init__.py`**

```python
```

(empty file)

- [ ] **Step 5: Write `src/learning_accelerator/observability/langfuse_setup.py`**

```python
from __future__ import annotations

import os

from langfuse.langchain import CallbackHandler


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_handler() -> CallbackHandler | None:
    if not langfuse_enabled():
        return None
    return CallbackHandler()


def get_langfuse_config(session_id: str) -> dict:
    config: dict = {"configurable": {"thread_id": session_id}}

    handler = get_langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
        config["metadata"] = {"langfuse_session_id": session_id}

    return config
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_langfuse_setup.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/learning_accelerator/observability/__init__.py src/learning_accelerator/observability/langfuse_setup.py tests/test_langfuse_setup.py
git commit -m "Add Langfuse setup module with graceful no-op when unconfigured"
```

---

### Task 2: .env.example and reference docker-compose file

**Files:**
- Modify: `.env.example`
- Create: `docker-compose.langfuse.yml`

**Interfaces:** none (config/reference files only)

- [ ] **Step 1: Add Langfuse vars to `.env.example`**

Append this section to `.env.example` (after the existing
`CHECKPOINT_DB_PATH` line):

```
# --- Langfuse observability (optional — leave blank to disable tracing) ---
# Use Langfuse Cloud (https://cloud.langfuse.com) or self-host via
# docker-compose.langfuse.yml in this repo.
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

- [ ] **Step 2: Write `docker-compose.langfuse.yml`**

This is a reference file for anyone who wants to run a real local
Langfuse instance — not executed as part of this chapter's own
verification (see the plan header's "Deliberate scope decision"). Base
it on Langfuse's own published self-host compose file:

```yaml
# Reference compose file for a local self-hosted Langfuse v3 instance.
# Not run automatically by this repo's tests or demo scripts — start it
# yourself with `docker compose -f docker-compose.langfuse.yml up -d` if
# you want real local tracing instead of Langfuse Cloud.
#
# CHANGEME-marked values are placeholders; replace them before any use
# beyond local experimentation.
services:
  langfuse-worker:
    image: langfuse/langfuse-worker:3
    restart: always
    depends_on: &langfuse-depends-on
      postgres:
        condition: service_healthy
      minio:
        condition: service_healthy
      redis:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
    ports:
      - 127.0.0.1:3030:3030
    environment: &langfuse-env
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/postgres
      SALT: "CHANGEME"
      ENCRYPTION_KEY: "0000000000000000000000000000000000000000000000000000000000000000" # CHANGEME: `openssl rand -hex 32`
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse # CHANGEME
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: minio
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: miniosecret # CHANGEME
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_AUTH: myredissecret # CHANGEME

  langfuse-web:
    image: langfuse/langfuse:3
    restart: always
    depends_on: *langfuse-depends-on
    ports:
      - 3000:3000
    environment:
      <<: *langfuse-env
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: CHANGEME

  postgres:
    image: postgres:16
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 3s
      timeout: 3s
      retries: 10
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    volumes:
      - langfuse_postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    restart: always
    entrypoint: sh
    command: -c 'mkdir -p /data/langfuse && minio server --address ":9000" --console-address ":9001" /data'
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: miniosecret
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 3s
      timeout: 3s
      retries: 10
    ports:
      - 9090:9000
    volumes:
      - langfuse_minio_data:/data

  redis:
    image: redis:7
    restart: always
    command: --requirepass myredissecret
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 3s
      timeout: 3s
      retries: 10

  clickhouse:
    image: clickhouse/clickhouse-server
    restart: always
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
    healthcheck:
      test: wget --no-verbose --tries=1 -O - http://localhost:8123/ping || exit 1
      interval: 3s
      timeout: 3s
      retries: 10
    volumes:
      - langfuse_clickhouse_data:/var/lib/clickhouse

volumes:
  langfuse_postgres_data:
  langfuse_minio_data:
  langfuse_clickhouse_data:
```

- [ ] **Step 3: Commit**

```bash
git add .env.example docker-compose.langfuse.yml
git commit -m "Add Langfuse env vars and reference self-host compose file"
```

---

### Task 3: Manual demo script + real verification (mock ingestion endpoint)

**Files:**
- Create: `scripts/demo_chapter6.py`

**Interfaces:**
- Consumes: `get_langfuse_config` (Task 1), `build_graph`/`initial_state`
  (chapters 2/5)

- [ ] **Step 1: Write `scripts/demo_chapter6.py`**

```python
"""Manual run: verify the Langfuse callback wiring fires real network
requests when configured, and gracefully omits callbacks when it isn't.

There's no live Langfuse instance in this environment — self-hosting the
full v3 stack (postgres+clickhouse+redis+minio+worker+web) is too heavy
to stand up just for this check (see docker-compose.langfuse.yml if you
want to run one yourself). Instead, this script runs a tiny local HTTP
server that stands in for Langfuse's ingestion endpoint and records
whether any POST requests arrive — proving our CallbackHandler wiring
genuinely fires real requests at LANGFUSE_HOST, not silently no-oping.

One real LLM call (Curriculum Planner) is enough to prove a traced call
fires; the graph then hits the existing human_approval interrupt, which
is a natural, fast stopping point for this specific check.

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly.
"""

from __future__ import annotations

import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from langfuse import get_client

from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import build_graph
from learning_accelerator.observability.langfuse_setup import get_langfuse_config

received_requests: list[tuple[str, int]] = []


class _MockIngestionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        received_requests.append((self.path, len(body)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"successes": [], "errors": []}')

    def log_message(self, format: str, *args: object) -> None:
        pass  # silence default per-request stderr logging


def main() -> None:
    server = HTTPServer(("localhost", 0), _MockIngestionHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test-chapter6"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test-chapter6"
    os.environ["LANGFUSE_HOST"] = f"http://localhost:{port}"

    session_id = str(uuid.uuid4())
    config = get_langfuse_config(session_id)
    assert "callbacks" in config, "expected Langfuse callback to be wired in when keys are set"

    graph = build_graph()
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)
    result = graph.invoke(state, config=config)
    assert "__interrupt__" in result, "expected the graph to stop at the human_approval interrupt"

    get_client().flush()
    server.shutdown()

    print(f"Mock Langfuse endpoint received {len(received_requests)} request(s).")
    for path, size in received_requests:
        print(f"  POST {path} ({size} bytes)")
    assert len(received_requests) > 0, "expected at least one request to reach the mock endpoint"

    print("\n--- now verifying graceful no-op when Langfuse is NOT configured ---")
    del os.environ["LANGFUSE_PUBLIC_KEY"]
    del os.environ["LANGFUSE_SECRET_KEY"]
    plain_config = get_langfuse_config(str(uuid.uuid4()))
    assert "callbacks" not in plain_config
    print("Confirmed: no 'callbacks' key when Langfuse credentials aren't set.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real against local Ollama**

```bash
rm -f .data/checkpoints.sqlite
LLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:12b uv run python scripts/demo_chapter6.py
```

Expected: prints "Mock Langfuse endpoint received N request(s)." with
N >= 1, prints at least one `POST /api/...` line, then the graceful
no-op confirmation line. This should run in well under a minute (a
single LLM call, not the multi-minute full loop from chapters 4-5).

- [ ] **Step 3: Run the full pytest suite once more**

Run: `uv run pytest -v`
Expected: all tests still pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_chapter6.py
git commit -m "Add chapter 6 manual demo script (mock Langfuse ingestion check)"
```

---

### Task 4: Chapter 6 learning flashcards

**Files:**
- Create: `learning/chapter6.md`

- [ ] **Step 1: Write `learning/chapter6.md`**

```markdown
## Chapter 6: Observability with Langfuse

**Q: How does Langfuse hook into a LangGraph run?**
A: Via LangChain's callback mechanism — `CallbackHandler` from
`langfuse.langchain` gets passed through `config["callbacks"]` at
`graph.invoke()` time, same as any other LangChain callback.

**Q: Why don't any of the existing agent node functions need to change
to get their LLM calls traced?**
A: LangGraph propagates the invoke-time `config` (including `callbacks`)
into every node's execution via Python contextvars, and LangChain's
`ChatModel.invoke()` reads that ambient config automatically through
`ensure_config()` when no explicit config is passed. Tracing is
transparent to node code.

**Q: How does a Langfuse trace get linked to our session/thread?**
A: Via the `metadata={"langfuse_session_id": session_id}` key in the
invoke config — `CallbackHandler` reads that specific metadata key and
maps it to Langfuse's own `session_id` trace attribute.

**Q: What happens if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't set?**
A: `get_langfuse_config()` omits the `callbacks` key entirely — the graph
runs exactly as it did in chapters 2-5, with zero tracing overhead and
no error. Tracing is opt-in, not required.

**Q: Why didn't this chapter stand up a real self-hosted Langfuse instance
to verify tracing?**
A: Langfuse's v3 self-host stack needs postgres + clickhouse + redis +
minio + two Langfuse services — too heavy to spin up just to check a
callback wiring. `docker-compose.langfuse.yml` is included as a reference
for anyone who wants to run one for real; verification instead used a
tiny local HTTP server standing in for the ingestion endpoint.

**Q: What did the mock-endpoint demo actually prove, and what does it NOT prove?**
A: It proves our integration code genuinely fires a real HTTP POST at
whatever `LANGFUSE_HOST` is configured to — the thing our code is
responsible for. It does NOT prove Langfuse's own server renders traces
correctly, since that's Langfuse's code, not ours.

**Q: Why does the demo only need to run through Curriculum Planner rather
than the full multi-topic loop?**
A: One real LLM call is enough to prove a callback fires. The graph
naturally stops at the existing `human_approval` interrupt right after,
which is a fast, convenient stopping point for this specific check.
```

- [ ] **Step 2: Commit**

```bash
git add learning/chapter6.md
git commit -m "Add chapter 6 learning flashcards"
```

---

### Task 5: Push, open, and merge the chapter 6 PR

**Files:** none (branch/PR operation only)

Full autonomy is authorized for chapters 5-9 — no confirmation needed
before push/PR/merge. Before this task, dispatch a final whole-branch
review (per subagent-driven-development) covering all of Tasks 1-4
together, and address any Critical/Important findings before proceeding.

- [ ] **Step 1: Confirm all tests pass**

Run: `uv run pytest -v`

- [ ] **Step 2: Push the branch**

```bash
git push -u origin agent/chapter-6
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "Chapter 6: Observability with Langfuse" --body "..."
```

(Compose the body summarizing what was built, referencing the mock
ingestion verification and the docker-compose.langfuse.yml reference
file, plus a test plan section — same style as chapters 1-5's PRs.)

- [ ] **Step 4: Merge**

```bash
gh pr merge --merge
```

- [ ] **Step 5: Delete the remote branch and sync the local worktree**

```bash
git push origin --delete agent/chapter-6
git checkout claude/agentic-ai-langgraph-course-ebf06b
git fetch origin
git merge --ff-only origin/main
git branch -d agent/chapter-6
```
