# Chapter 8: Cross-Framework Coordination with A2A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the Quiz Generator as a standalone A2A service, build a
CrewAI-based "Study Buddy" agent exposed the same way, wire the Progress
Coach to call the Study Buddy via A2A when a student scores low, and prove
the whole thing works with two real running servers and a real CrewAI
crew executed against local Ollama.

**Source material note:** same situation as chapter 7 — the article's own
page didn't yield chapter 8's text. The companion reference repo
(`github.com/sandeepmb/freecodecamp-multi-agent-ai-system`) has the actual
implementation (`src/a2a_services/`, `src/crewai_agent/`,
`tests/test_a2a.py`, `tests/test_crewai_interop.py`), pulled directly via
`gh api` and used as the structural basis here.

**Architecture:** `a2a_services/quiz_service.py` wraps this repo's own
`generate_questions`/`grade_answer` (chapter 4) in an A2A
`AgentExecutor`, served via `A2AStarletteApplication` + `uvicorn` on port
9001, discoverable at `/.well-known/agent-card.json`.
`a2a_services/a2a_client.py` provides generic `discover_agent`/`send_task`
plus two high-level helpers (`delegate_quiz_task`,
`request_study_assistance`) using JSON-RPC 2.0 `tasks/send`.
`crewai_agent/study_buddy.py` builds a one-agent, one-task CrewAI `Crew`
per request (fresh per task, no state leakage) using a custom
`TopicAnalyserTool`, served the same way on port 9002 — demonstrating that
a completely different framework (CrewAI, not LangGraph) can sit behind
the identical A2A protocol boundary. `agents/progress_coach.py` gains
`try_a2a_quiz_delegation` (available, tested, but — matching the
reference's own actual wiring — not automatically invoked by
`progress_coach_node`) and `try_study_buddy_assistance`, which IS wired
into `progress_coach_node`'s low-score path.

**Deliberate deviations from the reference repo:**
1. **A2A/CrewAI payloads stay plain dicts.** Unlike chapters 4/7 where
   Pydantic models added real type safety to in-process agent functions,
   A2A/CrewAI payloads cross a JSON-over-HTTP boundary — dicts are the
   natural representation here, matching the reference. `quiz_service.py`
   converts our Pydantic `QuizQuestion`/`GradeResult` (chapter 4) to dicts
   at the serialization boundary (`.model_dump()` / attribute access
   instead of the reference's dict `.get(...)` calls).
2. **Our `CoachingMessage` field is `tip`, not the reference's
   `encouragement`.** Already established in chapter 4, not renamed here.
3. **`try_a2a_quiz_delegation` is defined and tested but not wired into
   any graph node**, matching what the reference repo actually does (it's
   tested directly via `from agents.progress_coach import
   try_a2a_quiz_delegation` in `test_a2a.py`, but `progress_coach_node`'s
   body never calls it — only `try_study_buddy_assistance` is actually
   invoked, for the low-score path). Documented honestly in flashcards
   rather than silently "fixing" what might be intentional (a
   demonstration function, or a hook for future/manual use) by
   guessing at wiring the reference itself doesn't have.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-agentic-ai-course-9-chapters-design.md`
- Branch: `agent/chapter-8` (already created, off updated `main`)
- One PR per chapter; push, open, and merge it yourself once verified —
  full autonomy already authorized for chapters 5-9
- `tests/test_a2a.py` and `tests/test_crewai_interop.py` are fully mocked
  (no live server, no live LLM call, no live CrewAI execution) — matching
  the reference repo's own test suite exactly. These qualify as pure
  logic under this spec's testing policy (network/LLM calls are mocked at
  the boundary, e.g. `@patch("a2a_services.a2a_client.httpx.post")`) →
  full pytest coverage, NOT eval-marked, run in the default fast suite
- Real end-to-end verification (two real servers, real HTTP calls, a real
  CrewAI crew run against Ollama) is a manual-run demo script, per this
  spec's established policy for integration/LLM-calling code
- Quiz service: port 9001. Study Buddy: port 9002 (must differ — verified
  by a test in the reference, keep that test)
- Local dev machine has Ollama with `gemma4:12b-mlx`, `gemma4:12b`,
  `qwen3.5:2b`; chapters 2-7 all found `gemma4:12b` works reliably — use
  it as the default for both the LangGraph agents AND the CrewAI agent
  here (CrewAI's `LLM(model=f"ollama/{MODEL_NAME}", base_url=...)` talks
  to the same local Ollama instance)
- `learning/chapterN.md` format: short Q/A flashcards (see chapters 1-7)

---

### Task 1: A2A client helpers + tests

**Files:**
- Create: `src/learning_accelerator/a2a_services/__init__.py`
- Create: `src/learning_accelerator/a2a_services/a2a_client.py`
- Test: `tests/test_a2a_client.py`

**Interfaces:**
- Consumes: nothing
- Produces: `discover_agent`, `send_task`, `delegate_quiz_task`,
  `is_quiz_service_available`, `request_study_assistance`,
  `is_study_buddy_available` — used by `quiz_service.py`/`study_buddy.py`
  (Tasks 2-3, for their own health-checking if needed) and
  `agents/progress_coach.py` (Task 4)

- [ ] **Step 1: Add dependencies and verify import paths**

```bash
uv add a2a-sdk httpx
```

Then verify the actual importable module structure (the PyPI package is
`a2a-sdk`, but it's imported as `a2a`):

```bash
uv run python -c "from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, TextPart; print('ok: a2a.types')"
uv run python -c "from a2a.server.agent_execution import AgentExecutor, RequestContext; print('ok: agent_execution')"
uv run python -c "from a2a.server.apps import A2AStarletteApplication; print('ok: apps')"
uv run python -c "from a2a.server.events import EventQueue; print('ok: events')"
uv run python -c "from a2a.server.request_handlers import DefaultRequestHandler; print('ok: request_handlers')"
uv run python -c "from a2a.server.tasks import InMemoryTaskStore; print('ok: tasks')"
```

If any import fails, find the correct current path in the installed
version and use that instead — note in your report which paths you used.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_a2a_client.py`:

```python
import json

import httpx
import pytest
from unittest.mock import MagicMock, patch


class TestDiscoverAgent:
    @patch("learning_accelerator.a2a_services.a2a_client.httpx.get")
    def test_returns_card_on_success(self, mock_get):
        from learning_accelerator.a2a_services.a2a_client import discover_agent

        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "Test Agent", "url": "http://localhost:9001/"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = discover_agent("http://localhost:9001")
        assert result["name"] == "Test Agent"

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_connection_error(self, mock_get):
        from learning_accelerator.a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.ConnectError("Connection refused")
        assert discover_agent("http://localhost:9001") == {}

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_timeout(self, mock_get):
        from learning_accelerator.a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.TimeoutException("Timed out")
        assert discover_agent("http://localhost:9001") == {}


class TestSendTask:
    @patch("learning_accelerator.a2a_services.a2a_client.httpx.post")
    def test_returns_parsed_result_on_success(self, mock_post):
        from learning_accelerator.a2a_services.a2a_client import send_task

        quiz_result = {"status": "questions_ready", "topic": "Test Topic", "questions": []}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"artifacts": [{"parts": [{"type": "text", "text": json.dumps(quiz_result)}]}]}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = send_task("http://localhost:9001", json.dumps({"topic": "Test"}))
        assert result["status"] == "questions_ready"

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_connection_refused(self, mock_post):
        from learning_accelerator.a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.ConnectError("Connection refused")
        result = send_task("http://localhost:9001", "{}")
        assert "error" in result
        assert "connect" in result["error"].lower()

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_timeout(self, mock_post):
        from learning_accelerator.a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.TimeoutException("Timed out")
        result = send_task("http://localhost:9001", "{}", timeout=1.0)
        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestDelegateQuizTask:
    @patch("learning_accelerator.a2a_services.a2a_client.send_task")
    def test_sends_correct_payload(self, mock_send):
        from learning_accelerator.a2a_services.a2a_client import delegate_quiz_task

        mock_send.return_value = {"status": "graded", "score": 0.8}
        delegate_quiz_task(topic="LangGraph", explanation="Nodes and edges...", answers=["my answer"])

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["topic"] == "LangGraph"
        assert "explanation" in payload
        assert "answers" in payload

    @patch("learning_accelerator.a2a_services.a2a_client.send_task")
    def test_empty_answers_sends_empty_list(self, mock_send):
        from learning_accelerator.a2a_services.a2a_client import delegate_quiz_task

        mock_send.return_value = {"status": "questions_ready"}
        delegate_quiz_task("Topic", "Explanation", answers=None)

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["answers"] == []


class TestIsQuizServiceAvailable:
    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_returns_true_when_card_available(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_quiz_service_available

        mock_discover.return_value = {"name": "Quiz Service"}
        assert is_quiz_service_available() is True

    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_returns_false_when_service_down(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_quiz_service_available

        mock_discover.return_value = {}
        assert is_quiz_service_available() is False


class TestStudyBuddyClient:
    @patch("learning_accelerator.a2a_services.a2a_client.send_task")
    def test_request_study_assistance_sends_correct_payload(self, mock_send):
        from learning_accelerator.a2a_services.a2a_client import request_study_assistance

        mock_send.return_value = {"source": "crewai_study_buddy", "assistance": "...", "status": "complete"}
        request_study_assistance(topic="LangGraph", explanation="...", weak_areas=["checkpointing"])

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["topic"] == "LangGraph"
        assert payload["weak_areas"] == ["checkpointing"]
        assert "explanation" in payload

    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_is_study_buddy_available_true(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_study_buddy_available

        mock_discover.return_value = {"name": "CrewAI Study Buddy"}
        assert is_study_buddy_available() is True

    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_is_study_buddy_available_false(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_study_buddy_available

        mock_discover.return_value = {}
        assert is_study_buddy_available() is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_a2a_client.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'learning_accelerator.a2a_services'`

- [ ] **Step 4: Create `src/learning_accelerator/a2a_services/__init__.py`**

```python
```

(empty file)

- [ ] **Step 5: Write `src/learning_accelerator/a2a_services/a2a_client.py`**

```python
from __future__ import annotations

import json
import os
import uuid

import httpx

QUIZ_SERVICE_URL = os.getenv("QUIZ_SERVICE_URL", "http://localhost:9001")
STUDY_BUDDY_URL = os.getenv("STUDY_BUDDY_URL", "http://localhost:9002")
DEFAULT_TIMEOUT = 120.0


def discover_agent(base_url: str) -> dict:
    card_url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
    try:
        response = httpx.get(card_url, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[A2A Client] Cannot reach {card_url}: {e}")
        return {}


def send_task(
    base_url: str,
    message_text: str,
    task_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/send",
        "params": {
            "id": task_id or str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message_text}],
            },
        },
    }

    url = f"{base_url.rstrip('/')}/tasks/send"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        result = data.get("result", {})
        artifacts = result.get("artifacts", [])
        if artifacts:
            for part in artifacts[0].get("parts", []):
                if part.get("type") == "text":
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        return {"text": part["text"]}

        status = result.get("status", {})
        if status:
            msg = status.get("message", {})
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        return {"text": part["text"]}

        return result

    except httpx.TimeoutException:
        return {"error": f"Quiz service timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to quiz service at {url}"}
    except Exception as e:
        return {"error": f"A2A task failed: {type(e).__name__}: {e}"}


def delegate_quiz_task(
    topic: str,
    explanation: str,
    answers: list[str] | None = None,
    quiz_service_url: str = QUIZ_SERVICE_URL,
) -> dict:
    payload = json.dumps(
        {"topic": topic, "explanation": explanation, "answers": answers or []}
    )
    return send_task(quiz_service_url, payload)


def is_quiz_service_available(quiz_service_url: str = QUIZ_SERVICE_URL) -> bool:
    return bool(discover_agent(quiz_service_url))


def request_study_assistance(
    topic: str,
    explanation: str,
    weak_areas: list[str] | None = None,
    study_buddy_url: str = STUDY_BUDDY_URL,
) -> dict:
    payload = json.dumps(
        {"topic": topic, "explanation": explanation, "weak_areas": weak_areas or []}
    )
    return send_task(study_buddy_url, payload, timeout=180.0)


def is_study_buddy_available(study_buddy_url: str = STUDY_BUDDY_URL) -> bool:
    return bool(discover_agent(study_buddy_url))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_a2a_client.py -v`
Expected: 11 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/learning_accelerator/a2a_services/__init__.py src/learning_accelerator/a2a_services/a2a_client.py tests/test_a2a_client.py
git commit -m "Add A2A client helpers (discover, send_task, quiz/study-buddy delegation)"
```

---

### Task 2: Quiz Generator as a standalone A2A service + tests

**Files:**
- Create: `src/learning_accelerator/a2a_services/quiz_service.py`
- Test: `tests/test_quiz_service.py`

**Interfaces:**
- Consumes: `generate_questions`/`grade_answer` (chapter 4, returns
  `QuizQuestion`/`GradeResult` Pydantic models)
- Produces: `QUIZ_AGENT_CARD`, `QuizAgentExecutor`, `create_quiz_server()`
  — run for real in Task 5's demo

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quiz_service.py`:

```python
class TestQuizAgentCard:
    def test_agent_card_has_required_fields(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.name
        assert QUIZ_AGENT_CARD.url
        assert QUIZ_AGENT_CARD.version
        assert QUIZ_AGENT_CARD.skills
        assert len(QUIZ_AGENT_CARD.skills) > 0

    def test_agent_card_url_is_port_9001(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert "9001" in QUIZ_AGENT_CARD.url

    def test_skill_has_required_fields(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        skill = QUIZ_AGENT_CARD.skills[0]
        assert skill.id
        assert skill.name
        assert skill.description
        assert len(skill.description) > 20

    def test_skill_has_examples(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        skill = QUIZ_AGENT_CARD.skills[0]
        assert skill.examples
        assert len(skill.examples) >= 1

    def test_skill_id_is_correct(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.skills[0].id == "generate_and_grade_quiz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quiz_service.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError`

- [ ] **Step 3: Write `src/learning_accelerator/a2a_services/quiz_service.py`**

```python
"""
Quiz Generator exposed as a standalone A2A service.

Run standalone:
  uv run python src/learning_accelerator/a2a_services/quiz_service.py

Then discover:
  curl http://localhost:9001/.well-known/agent-card.json
"""

from __future__ import annotations

import asyncio
import json

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, TextPart

from learning_accelerator.agents.quiz_generator import generate_questions, grade_answer

QUIZ_SKILL = AgentSkill(
    id="generate_and_grade_quiz",
    name="Generate and Grade Quiz",
    description=(
        "Given a topic and optional explanation text, generates quiz questions "
        "that test conceptual understanding. If answers are provided, grades "
        "each answer and returns scores with identified weak areas."
    ),
    tags=["quiz", "assessment", "education", "grading"],
    examples=[
        "Generate a quiz on LangGraph state management",
        "Grade these answers for a checkpointing quiz: ...",
    ],
)

QUIZ_AGENT_CARD = AgentCard(
    name="Quiz Generator Service",
    description=(
        "A specialised quiz generation and grading service built with LangGraph. "
        "Generates questions that test genuine understanding, grades answers "
        "using LLM-as-judge, and identifies weak areas for further study. "
        "Framework-agnostic: works with any A2A-compatible agent."
    ),
    url="http://localhost:9001/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[QUIZ_SKILL],
)


class QuizAgentExecutor(AgentExecutor):
    """Handles incoming A2A quiz tasks.

    Request format (JSON in the text part):
    {"topic": "...", "explanation": "...", "answers": [...]}   (answers optional)

    Response format (JSON in the text part):
    {"status": "questions_ready" | "graded", "topic": ..., "questions": [...],
     "score": ..., "graded_questions": [...], "weak_areas": [...]}
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = ""
        for part in context.current_request.params.message.parts:
            if isinstance(part, TextPart):
                request_text += part.text

        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError:
            request_data = {"topic": request_text, "explanation": ""}

        topic = request_data.get("topic", "General Knowledge")
        explanation = request_data.get("explanation", "")
        provided_answers = request_data.get("answers", [])

        print(
            f"[Quiz A2A] Task received: topic='{topic}', "
            f"answers_provided={len(provided_answers)}"
        )

        questions = await asyncio.to_thread(generate_questions, topic, explanation, 3)
        questions_data = [q.model_dump() for q in questions]

        if not provided_answers:
            result = {
                "status": "questions_ready",
                "topic": topic,
                "questions": questions_data,
                "message": "Questions generated. Submit again with 'answers' key to grade.",
            }
        else:
            graded = []
            total_score = 0.0
            weak_areas: list[str] = []

            for q, answer in zip(questions, provided_answers):
                grade = await asyncio.to_thread(
                    grade_answer, q.question, q.expected_answer, answer
                )
                total_score += grade.score
                if grade.missing_concept:
                    weak_areas.append(grade.missing_concept)

                graded.append(
                    {
                        "question": q.question,
                        "answer": answer,
                        "score": grade.score,
                        "correct": grade.correct,
                        "feedback": grade.feedback,
                    }
                )

            avg_score = total_score / len(questions) if questions else 0.0

            result = {
                "status": "graded",
                "topic": topic,
                "score": avg_score,
                "questions": questions_data,
                "graded_questions": graded,
                "weak_areas": list(set(weak_areas)),
            }

        print(f"[Quiz A2A] Task complete: status={result['status']}")

        await event_queue.enqueue_event(
            Message(role="agent", parts=[TextPart(text=json.dumps(result, indent=2))])
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def create_quiz_server():
    request_handler = DefaultRequestHandler(
        agent_executor=QuizAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=QUIZ_AGENT_CARD, http_handler=request_handler)
    return app.build()


if __name__ == "__main__":
    print("[Quiz A2A Service] Starting on http://localhost:9001")
    print("[Quiz A2A Service] Agent Card: http://localhost:9001/.well-known/agent-card.json")
    print("[Quiz A2A Service] Press Ctrl+C to stop\n")
    uvicorn.run(create_quiz_server(), host="0.0.0.0", port=9001, log_level="warning")
```

(If Task 1's import-path verification found different paths work in the
installed `a2a-sdk` version, use those instead here — keep consistent
with what Task 1 established.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quiz_service.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/learning_accelerator/a2a_services/quiz_service.py tests/test_quiz_service.py
git commit -m "Add Quiz Generator as a standalone A2A service"
```

---

### Task 3: CrewAI Study Buddy as an A2A service + tests

**Files:**
- Create: `src/learning_accelerator/crewai_agent/__init__.py`
- Create: `src/learning_accelerator/crewai_agent/study_buddy.py`
- Test: `tests/test_crewai_interop.py`

**Interfaces:**
- Consumes: nothing from this repo's own agents (deliberately —
  demonstrates a fully independent framework)
- Produces: `STUDY_BUDDY_CARD`, `TopicAnalyserTool`,
  `build_study_buddy_crew()`, `StudyBuddyExecutor`,
  `create_study_buddy_server()` — run for real in Task 5's demo

- [ ] **Step 1: Add the crewai dependency**

```bash
uv add crewai
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_crewai_interop.py`:

```python
import json


class TestStudyBuddyAgentCard:
    def test_agent_card_has_required_fields(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.name
        assert STUDY_BUDDY_CARD.url
        assert STUDY_BUDDY_CARD.version
        assert STUDY_BUDDY_CARD.skills
        assert len(STUDY_BUDDY_CARD.skills) > 0

    def test_agent_card_url_is_port_9002(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert "9002" in STUDY_BUDDY_CARD.url

    def test_skill_id_is_correct(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.skills[0].id == "supplementary_study_assistance"

    def test_skill_mentions_crewai(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        card_text = (STUDY_BUDDY_CARD.description + STUDY_BUDDY_CARD.skills[0].description).lower()
        assert "crewai" in card_text

    def test_different_port_from_quiz_service(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.url != QUIZ_AGENT_CARD.url


class TestBuildStudyBuddyCrew:
    def test_returns_crew_object(self):
        from crewai import Crew

        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="LangGraph Checkpointing",
            explanation="Checkpoints persist state after each step...",
            weak_areas=["thread_id"],
        )
        assert isinstance(crew, Crew)

    def test_crew_has_one_agent(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        assert len(crew.agents) == 1

    def test_crew_has_one_task(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        assert len(crew.tasks) == 1

    def test_agent_has_study_buddy_role(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        agent = crew.agents[0]
        assert "study" in agent.role.lower() or "buddy" in agent.role.lower()

    def test_task_description_contains_topic(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="LangGraph Reducers", explanation="Reducers control merging...", weak_areas=[]
        )
        assert "LangGraph Reducers" in crew.tasks[0].description

    def test_task_description_contains_weak_areas(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="Checkpointing", explanation="...", weak_areas=["thread_id", "SqliteSaver"]
        )
        desc = crew.tasks[0].description
        assert "thread_id" in desc or "SqliteSaver" in desc

    def test_agent_has_topic_analyser_tool(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        tool_names = [type(t).__name__ for t in crew.agents[0].tools]
        assert "TopicAnalyserTool" in tool_names

    def test_different_topics_create_different_tasks(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew1 = build_study_buddy_crew("Checkpointing", "Explanation 1", [])
        crew2 = build_study_buddy_crew("Reducers", "Explanation 2", [])
        assert crew1.tasks[0].description != crew2.tasks[0].description


class TestTopicAnalyserTool:
    def test_returns_json_string(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = tool._run(topic="LangGraph Checkpointing", weak_areas=["thread_id"])
        assert isinstance(json.loads(result), dict)

    def test_result_contains_topic(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Reducers", weak_areas=[]))
        assert result["topic"] == "Reducers"

    def test_result_has_required_keys(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Checkpointing", weak_areas=["thread_id"]))
        for key in ["topic", "focus_areas", "suggested_approach", "study_tip"]:
            assert key in result, f"Missing key: {key}"

    def test_weak_areas_appear_in_focus_areas(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(
            tool._run(topic="Checkpointing", weak_areas=["thread_id", "SqliteSaver"])
        )
        assert "thread_id" in result["focus_areas"]
        assert "SqliteSaver" in result["focus_areas"]

    def test_empty_weak_areas_uses_fallback(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="LangGraph Basics", weak_areas=[]))
        assert len(result["focus_areas"]) > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_crewai_interop.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError`

- [ ] **Step 4: Create `src/learning_accelerator/crewai_agent/__init__.py`**

```python
```

(empty file)

- [ ] **Step 5: Write `src/learning_accelerator/crewai_agent/study_buddy.py`**

```python
"""
A CrewAI-based study buddy agent exposed as an A2A service.

Demonstrates cross-framework agent interoperability: built with CrewAI
(not LangGraph), exposed via the same A2A protocol as the Quiz Service,
callable by the LangGraph Progress Coach without either framework knowing
about the other's internals.

Run standalone:
  uv run python src/learning_accelerator/crewai_agent/study_buddy.py

Agent Card:
  http://localhost:9002/.well-known/agent-card.json
"""

from __future__ import annotations

import asyncio
import json
import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, TextPart
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class TopicAnalyserInput(BaseModel):
    topic: str = Field(description="The topic to analyse")
    weak_areas: list[str] = Field(
        default_factory=list, description="Weak areas the student struggled with"
    )


class TopicAnalyserTool(BaseTool):
    """Analyses a topic and weak areas to produce a structured study plan."""

    name: str = "topic_analyser"
    description: str = (
        "Analyse a study topic and the student's weak areas to produce "
        "a structured list of key concepts to focus on."
    )
    args_schema: type[BaseModel] = TopicAnalyserInput

    def _run(self, topic: str, weak_areas: list[str] | None = None) -> str:
        areas = weak_areas or []
        focus_items = areas if areas else [f"Core concepts of {topic}"]

        analysis = {
            "topic": topic,
            "focus_areas": focus_items,
            "suggested_approach": (
                f"Start with the fundamentals of {topic}, then address: "
                f"{', '.join(focus_items)}."
            ),
            "study_tip": (
                "Try explaining the concept out loud in your own words. "
                "If you can teach it simply, you understand it."
            ),
        }
        return json.dumps(analysis)


def build_study_buddy_crew(topic: str, explanation: str, weak_areas: list[str]) -> Crew:
    """Build a fresh CrewAI crew for one A2A task (no state leakage between tasks)."""
    topic_analyser = TopicAnalyserTool()

    llm = LLM(model=f"ollama/{MODEL_NAME}", base_url=OLLAMA_BASE_URL)

    study_buddy_agent = Agent(
        role="Study Buddy",
        goal=(
            "Provide clear, encouraging supplementary explanations that help "
            "students understand difficult concepts from a fresh angle."
        ),
        backstory=(
            "You are an experienced tutor who has helped hundreds of students "
            "master LangGraph and agentic AI concepts. You specialise in finding "
            "alternative explanations and analogies that make difficult ideas click."
        ),
        llm=llm,
        tools=[topic_analyser],
        verbose=False,
        allow_delegation=False,
    )

    weak_areas_text = (
        f"The student struggled with: {', '.join(weak_areas)}"
        if weak_areas
        else "No specific weak areas identified."
    )

    study_task = Task(
        description=(
            f"A student is studying '{topic}'. Here is the explanation they received:\n\n"
            f"{explanation[:1000]}\n\n"
            f"{weak_areas_text}\n\n"
            "First use the topic_analyser tool to structure your approach. "
            "Then provide: "
            "1) A fresh analogy that explains the core concept differently, "
            "2) One concrete example that illustrates the weak area(s), "
            "3) One practical tip for remembering this concept. "
            "Keep your response concise and encouraging (150-250 words)."
        ),
        agent=study_buddy_agent,
        expected_output=(
            "A structured study assistance response with a fresh analogy, "
            "a concrete example targeting weak areas, and a memory tip."
        ),
    )

    return Crew(
        agents=[study_buddy_agent],
        tasks=[study_task],
        process=Process.sequential,
        verbose=False,
    )


STUDY_BUDDY_SKILL = AgentSkill(
    id="supplementary_study_assistance",
    name="Supplementary Study Assistance",
    description=(
        "Provides supplementary study assistance when a student needs a "
        "different explanation angle. Given a topic, the original explanation, "
        "and any weak areas, returns a fresh analogy, a targeted example, "
        "and a memory tip. Built with CrewAI."
    ),
    tags=["study", "tutoring", "explanation", "crewai"],
    examples=[
        "Help a student understand LangGraph checkpointing from a different angle",
        "Provide supplementary explanation for reducer weak areas",
    ],
)

STUDY_BUDDY_CARD = AgentCard(
    name="CrewAI Study Buddy",
    description=(
        "A supplementary learning assistant built with CrewAI. Provides "
        "alternative explanations and targeted examples when the primary "
        "explanation didn't land. Framework-agnostic: connects via A2A protocol."
    ),
    url="http://localhost:9002/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[STUDY_BUDDY_SKILL],
)


class StudyBuddyExecutor(AgentExecutor):
    """Bridges the A2A protocol to CrewAI execution."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = ""
        for part in context.current_request.params.message.parts:
            if isinstance(part, TextPart):
                request_text += part.text

        try:
            request_data = json.loads(request_text)
        except json.JSONDecodeError:
            request_data = {"topic": request_text}

        topic = request_data.get("topic", "General Topic")
        explanation = request_data.get("explanation", "")
        weak_areas = request_data.get("weak_areas", [])

        print(f"[Study Buddy A2A] Request: topic='{topic}', weak_areas={weak_areas}")

        try:
            crew = build_study_buddy_crew(topic, explanation, weak_areas)
            crew_result = await asyncio.to_thread(crew.kickoff)

            result_text = str(crew_result)
            if hasattr(crew_result, "raw"):
                result_text = crew_result.raw

            result = {
                "source": "crewai_study_buddy",
                "topic": topic,
                "weak_areas": weak_areas,
                "assistance": result_text,
                "status": "complete",
            }
            print(f"[Study Buddy A2A] Task complete ({len(result_text)} chars)")

        except Exception as e:
            print(f"[Study Buddy A2A] CrewAI error: {e}")
            result = {
                "source": "crewai_study_buddy",
                "topic": topic,
                "assistance": (
                    "I encountered an issue generating supplementary help for "
                    f"'{topic}'. Please review the original explanation and try again."
                ),
                "status": "error",
                "error": str(e),
            }

        await event_queue.enqueue_event(
            Message(role="agent", parts=[TextPart(text=json.dumps(result, indent=2))])
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def create_study_buddy_server():
    request_handler = DefaultRequestHandler(
        agent_executor=StudyBuddyExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=STUDY_BUDDY_CARD, http_handler=request_handler)
    return app.build()


if __name__ == "__main__":
    print("[CrewAI Study Buddy] Starting on http://localhost:9002")
    print("[CrewAI Study Buddy] Agent Card: http://localhost:9002/.well-known/agent-card.json")
    print("[CrewAI Study Buddy] This is a CrewAI agent served via A2A")
    print("[CrewAI Study Buddy] Press Ctrl+C to stop\n")
    uvicorn.run(create_study_buddy_server(), host="0.0.0.0", port=9002, log_level="warning")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_crewai_interop.py -v`
Expected: 15 passed (none of these tests execute `crew.kickoff()` — they
only inspect the crew's structure, so no live LLM call happens here)

- [ ] **Step 7: Run the full default suite to confirm no regressions**

Run: `uv run pytest -v`

- [ ] **Step 8: Commit**

```bash
git add src/learning_accelerator/crewai_agent/__init__.py src/learning_accelerator/crewai_agent/study_buddy.py tests/test_crewai_interop.py
git commit -m "Add CrewAI Study Buddy as a cross-framework A2A service"
```

---

### Task 4: Wire A2A delegation into Progress Coach

**Files:**
- Modify: `src/learning_accelerator/agents/progress_coach.py`
- Test: `tests/test_progress_coach_a2a.py`

**Interfaces:**
- Consumes: `a2a_client` functions (Task 1)
- Produces: `try_a2a_quiz_delegation(topic, explanation, answers) -> dict
  | None` (defined, tested, NOT wired into `progress_coach_node` — see
  plan header), `try_study_buddy_assistance(topic, explanation,
  weak_areas) -> str | None` (wired into `progress_coach_node`'s
  low-score path)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress_coach_a2a.py`:

```python
import os
from unittest.mock import patch


class TestTryA2AQuizDelegation:
    def test_a2a_disabled_returns_none(self):
        with patch.dict(os.environ, {"USE_A2A_QUIZ": "false"}):
            from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

            assert try_a2a_quiz_delegation("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_quiz_service_available", return_value=False)
    def test_returns_none_when_service_unavailable(self, mock_available):
        from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

        assert try_a2a_quiz_delegation("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_quiz_service_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.delegate_quiz_task")
    def test_returns_result_when_service_available(self, mock_delegate, mock_available):
        from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

        mock_delegate.return_value = {"status": "graded", "score": 0.85, "weak_areas": []}
        result = try_a2a_quiz_delegation("Topic", "Explanation", ["answer1"])
        assert result is not None
        assert result["status"] == "graded"

    @patch("learning_accelerator.a2a_services.a2a_client.is_quiz_service_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.delegate_quiz_task")
    def test_returns_none_on_delegation_error(self, mock_delegate, mock_available):
        from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

        mock_delegate.return_value = {"error": "Service crashed"}
        assert try_a2a_quiz_delegation("Topic", "Explanation", []) is None


class TestTryStudyBuddyAssistance:
    def test_disabled_by_env_var_returns_none(self):
        with patch.dict(os.environ, {"USE_STUDY_BUDDY": "false"}):
            from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

            assert try_study_buddy_assistance("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_study_buddy_available", return_value=False)
    def test_returns_none_when_service_down(self, mock_avail):
        from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

        assert try_study_buddy_assistance("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_study_buddy_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.request_study_assistance")
    def test_returns_assistance_text_when_available(self, mock_assist, mock_avail):
        from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

        mock_assist.return_value = {
            "source": "crewai_study_buddy",
            "assistance": "Think of a checkpoint like a save file...",
            "status": "complete",
        }
        result = try_study_buddy_assistance("Topic", "Explanation", ["thread_id"])
        assert result is not None
        assert "save file" in result

    @patch("learning_accelerator.a2a_services.a2a_client.is_study_buddy_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.request_study_assistance")
    def test_returns_none_on_error_response(self, mock_assist, mock_avail):
        from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

        mock_assist.return_value = {"status": "error", "error": "CrewAI crashed"}
        assert try_study_buddy_assistance("Topic", "Explanation", []) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_progress_coach_a2a.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'try_a2a_quiz_delegation'`

- [ ] **Step 3: Edit `src/learning_accelerator/agents/progress_coach.py`**

Add near the top (after existing imports), env-driven URLs read at
**call time** inside each function (not module load time — this matters
for tests that patch `os.environ` per-test):

```python
QUIZ_SERVICE_URL = "http://localhost:9001"
STUDY_BUDDY_URL = "http://localhost:9002"
```

Add these two functions (place them before `progress_coach_node`):

```python
def try_a2a_quiz_delegation(
    topic: str, explanation: str, answers: list[str]
) -> dict | None:
    """Attempt to delegate quiz grading to the A2A Quiz Service.

    Returns the grading result dict if successful, None if the service is
    disabled, unavailable, or returns an error — callers should fall back
    to local quiz generation in that case. Not currently called by
    progress_coach_node (see learning/chapter8.md for why); available for
    direct/manual use.
    """
    use_a2a = os.environ.get("USE_A2A_QUIZ", "true").lower() == "true"
    if not use_a2a:
        return None

    from learning_accelerator.a2a_services.a2a_client import (
        delegate_quiz_task,
        is_quiz_service_available,
    )

    quiz_service_url = os.environ.get("QUIZ_SERVICE_URL", QUIZ_SERVICE_URL)

    if not is_quiz_service_available(quiz_service_url):
        print(
            f"[Progress Coach] Quiz A2A service not available at "
            f"{quiz_service_url}, using local quiz generator"
        )
        return None

    print(f"[Progress Coach] Delegating quiz to A2A service: {quiz_service_url}")
    result = delegate_quiz_task(
        topic=topic,
        explanation=explanation,
        answers=answers,
        quiz_service_url=quiz_service_url,
    )

    if "error" in result:
        print(f"[Progress Coach] A2A delegation failed: {result['error']}")
        return None

    print(f"[Progress Coach] A2A quiz complete: score={result.get('score', 0):.0%}")
    return result


def try_study_buddy_assistance(
    topic: str, explanation: str, weak_areas: list[str]
) -> str | None:
    """Request supplementary study help from the CrewAI Study Buddy.

    Called when a student scores below the pass threshold. Returns the
    assistance text if available, None if the service is disabled,
    unavailable, or returns an error.
    """
    use_study_buddy = os.environ.get("USE_STUDY_BUDDY", "true").lower() == "true"
    if not use_study_buddy:
        return None

    from learning_accelerator.a2a_services.a2a_client import (
        is_study_buddy_available,
        request_study_assistance,
    )

    study_buddy_url = os.environ.get("STUDY_BUDDY_URL", STUDY_BUDDY_URL)

    if not is_study_buddy_available(study_buddy_url):
        return None

    print("[Progress Coach] Requesting study assistance from CrewAI Study Buddy...")
    result = request_study_assistance(
        topic=topic,
        explanation=explanation,
        weak_areas=weak_areas,
        study_buddy_url=study_buddy_url,
    )

    if "error" in result or result.get("status") == "error":
        return None

    return result.get("assistance", "")
```

Then wire `try_study_buddy_assistance` into `progress_coach_node`. Change:

```python
    return {
        "roadmap": roadmap,
        "current_topic_index": idx + 1,
        "messages": [AIMessage(content=coaching.summary)],
        "error": None,
    }
```

to (add the Study Buddy call before the final `return`, mirroring where
the reference calls it — after coaching/status update, only when the
score is below threshold and there are weak areas to address):

```python
    if latest.score < PASS_THRESHOLD and latest.weak_areas:
        assistance = try_study_buddy_assistance(
            topic=latest.topic,
            explanation="",
            weak_areas=latest.weak_areas,
        )
        if assistance:
            print("\n" + "─" * 60)
            print("Study Buddy (via CrewAI → A2A):")
            print(assistance)
            print("─" * 60 + "\n")

    return {
        "roadmap": roadmap,
        "current_topic_index": idx + 1,
        "messages": [AIMessage(content=coaching.summary)],
        "error": None,
    }
```

(`explanation=""` here — unlike the reference, this codebase's
`progress_coach_node` doesn't currently extract the Explainer's last
message from `state["messages"]`; adding that is a reasonable
enhancement, but do it only if it's a small, clearly-scoped addition —
otherwise leave the explanation blank and note this as a known
simplification in the flashcards. Your call; document whichever you do.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_progress_coach_a2a.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the full default suite**

Run: `uv run pytest -v`
Expected: all tests pass (existing `test_progress_coach.py` tests must
still pass unchanged — this task only adds new functions and one
conditional block, it doesn't change `next_topic_status`/`route_after_coach`)

- [ ] **Step 6: Commit**

```bash
git add src/learning_accelerator/agents/progress_coach.py tests/test_progress_coach_a2a.py
git commit -m "Wire CrewAI Study Buddy A2A delegation into Progress Coach's low-score path"
```

---

### Task 5: Manual demo script + real end-to-end verification

**Files:**
- Create: `scripts/demo_chapter8.py`

**Interfaces:**
- Consumes: `create_quiz_server`/`create_study_buddy_server` (Tasks 2-3),
  `a2a_client` functions (Task 1)

This is the real test of cross-framework coordination: two actual HTTP
servers, a real CrewAI crew executed against local Ollama, real A2A
JSON-RPC calls between them.

- [ ] **Step 1: Write `scripts/demo_chapter8.py`**

```python
"""Manual run: start both A2A services for real, then call each one over
real HTTP — proving genuine cross-framework coordination (a LangGraph
service and a CrewAI service, both behind the same A2A protocol).

Requires local Ollama running (OLLAMA_MODEL, default gemma4:12b) — used
by both the Quiz Generator's LangChain calls and the CrewAI Study Buddy's
LLM calls.
"""

from __future__ import annotations

import multiprocessing
import time

import uvicorn

from learning_accelerator.a2a_services.a2a_client import (
    delegate_quiz_task,
    discover_agent,
    request_study_assistance,
)


def _run_quiz_server() -> None:
    from learning_accelerator.a2a_services.quiz_service import create_quiz_server

    uvicorn.run(create_quiz_server(), host="127.0.0.1", port=9001, log_level="warning")


def _run_study_buddy_server() -> None:
    from learning_accelerator.crewai_agent.study_buddy import create_study_buddy_server

    uvicorn.run(create_study_buddy_server(), host="127.0.0.1", port=9002, log_level="warning")


def _wait_until_healthy(url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if discover_agent(url):
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    quiz_process = multiprocessing.Process(target=_run_quiz_server, daemon=True)
    study_buddy_process = multiprocessing.Process(target=_run_study_buddy_server, daemon=True)

    quiz_process.start()
    study_buddy_process.start()

    try:
        print("Waiting for both A2A services to come up...")
        assert _wait_until_healthy("http://localhost:9001"), "Quiz service never became healthy"
        assert _wait_until_healthy("http://localhost:9002"), "Study Buddy service never became healthy"

        quiz_card = discover_agent("http://localhost:9001")
        study_buddy_card = discover_agent("http://localhost:9002")
        print(f"Quiz service card: {quiz_card['name']} (skills: {[s['id'] for s in quiz_card['skills']]})")
        print(f"Study Buddy card: {study_buddy_card['name']} (skills: {[s['id'] for s in study_buddy_card['skills']]})")

        print("\n--- Delegating quiz generation to the Quiz A2A service ---")
        quiz_result = delegate_quiz_task(
            topic="LangGraph Checkpointing",
            explanation=(
                "A checkpoint is a saved snapshot of the graph's state after a "
                "step, persisted so a run can be paused, resumed, or recovered "
                "after a crash."
            ),
        )
        print(f"Quiz result status: {quiz_result.get('status')}")
        assert quiz_result.get("status") == "questions_ready", f"Unexpected quiz result: {quiz_result}"
        print(f"Questions generated: {len(quiz_result.get('questions', []))}")

        print("\n--- Requesting supplementary help from the CrewAI Study Buddy (real crew.kickoff()) ---")
        assistance = request_study_assistance(
            topic="LangGraph Checkpointing",
            explanation=(
                "A checkpoint is a saved snapshot of the graph's state after a "
                "step, persisted so a run can be paused, resumed, or recovered "
                "after a crash."
            ),
            weak_areas=["thread_id", "SqliteSaver"],
        )
        print(f"Study Buddy status: {assistance.get('status')}")
        assert assistance.get("status") == "complete", f"Unexpected Study Buddy result: {assistance}"
        print(f"Assistance text ({len(assistance.get('assistance', ''))} chars):")
        print(assistance.get("assistance", ""))

    finally:
        quiz_process.terminate()
        study_buddy_process.terminate()
        quiz_process.join(timeout=5)
        study_buddy_process.join(timeout=5)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real against local Ollama**

This starts two real servers and runs a real CrewAI crew — expect
several minutes total (CrewAI's own LLM calls plus this repo's quiz
generation). Run in the background rather than blocking.

```bash
rm -f .data/checkpoints.sqlite
OLLAMA_MODEL=gemma4:12b uv run python scripts/demo_chapter8.py
```

Expected: both agent cards print with the correct names/skills, the quiz
service returns `"questions_ready"` with real generated questions, and
the CrewAI Study Buddy returns `"complete"` with real assistance text. If
CrewAI's Ollama integration errors out or hangs, check whether `crewai`
pulled in a version of `litellm` that needs a different `ollama/` model
string format, and adjust — note whatever you find in your report.

- [ ] **Step 3: Run the full pytest suite once more**

Run: `uv run pytest -v`
Expected: all tests still pass (this step only adds a script).

- [ ] **Step 4: Commit**

```bash
git add scripts/demo_chapter8.py
git commit -m "Add chapter 8 manual demo script (real cross-framework A2A run)"
```

---

### Task 6: Chapter 8 learning flashcards

**Files:**
- Create: `learning/chapter8.md`

- [ ] **Step 1: Write `learning/chapter8.md`**

Cover at minimum: what A2A solves and why it's protocol-based rather than
framework-specific adapters, what an Agent Card is and where it's served,
why the Quiz Service converts our Pydantic models to dicts at the
serialization boundary, why `try_a2a_quiz_delegation` exists but isn't
wired into any graph node (an honest description of what was found in
the reference, not a guess dressed up as certainty), how the CrewAI
Study Buddy proves genuine cross-framework interop, and the real observed
results from Task 5's live run (fill in with actual output — don't leave
placeholder text).

- [ ] **Step 2: Commit**

```bash
git add learning/chapter8.md
git commit -m "Add chapter 8 learning flashcards"
```

---

### Task 7: Push, open, and merge the chapter 8 PR

**Files:** none (branch/PR operation only)

Full autonomy is authorized for chapters 5-9 — no confirmation needed
before push/PR/merge. Before this task, dispatch a final whole-branch
review (per subagent-driven-development) covering all of Tasks 1-6
together, and address any Critical/Important findings before proceeding.

- [ ] **Step 1: Confirm the default fast suite passes**

Run: `uv run pytest -v`

- [ ] **Step 2: Push the branch**

```bash
git push -u origin agent/chapter-8
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "Chapter 8: Cross-Framework Coordination with A2A" --body "..."
```

- [ ] **Step 4: Merge**

```bash
gh pr merge --merge
```

- [ ] **Step 5: Delete the remote branch and sync the local worktree**

```bash
git push origin --delete agent/chapter-8
git checkout claude/agentic-ai-langgraph-course-ebf06b
git fetch origin
git merge --ff-only origin/main
git branch -d agent/chapter-8
```
