# Node Function Test Coverage

## Goal

None of the four LangGraph node functions (`curriculum_planner_node`,
`explainer_node`, `quiz_generator_node`, `progress_coach_node`) have a
fast-suite unit test exercising the node wrapper itself — confirmed by
grepping the test suite: none are called by name in any test file except
`explainer_node`, which is only exercised by `test_eval.py`, a real-LLM
test excluded from the default `pytest -q` run via the `eval` marker.
Their sub-functions (`generate_questions`, `grade_answer`,
`get_coaching_message`, `try_a2a_quiz_delegation`,
`try_study_buddy_assistance`, `next_topic_status`, `route_after_coach`)
already have solid unit test coverage — this closes the remaining gap:
the node wrappers' own state read/write shape, error handling, and how
they orchestrate their sub-functions.

This is the first of two approved follow-ups from the runtime
performance pass (PR #19); the second (a Streamlit `@st.cache_resource`
fix) is a separate, later piece of work.

## File structure

- `tests/test_curriculum_planner.py` (new)
- `tests/test_explainer.py` (new)
- `tests/test_quiz_generator.py` (new)
- `tests/test_progress_coach.py` (extended — already has
  `next_topic_status`/`route_after_coach` tests; `progress_coach_node`
  tests are added alongside them)

## Mocking pattern

All mocking uses `unittest.mock.MagicMock` — already the established
mocking library across this suite (`test_a2a_client.py`,
`test_judge_model.py`, `test_streamlit_app.py`), no new dependency.

The mock boundary is drawn at the nearest already-tested collaborator,
not uniformly at `get_chat_model`:

- **`curriculum_planner_node`** and **`explainer_node`** mock
  `get_chat_model` directly (`monkeypatch.setattr(<module>,
  "get_chat_model", MagicMock(return_value=fake_llm))`, with
  `fake_llm.with_structured_output.return_value` / `.bind_tools.return_value`
  another `MagicMock` whose `.invoke` is set via `return_value` or
  `side_effect`) — there is no lower boundary to mock, since these
  functions *are* the LLM call/loop being tested. This exactly follows
  the pattern already used in `tests/test_judge_model.py`.
- **`quiz_generator_node`** mocks `run_quiz` directly
  (`monkeypatch.setattr(quiz_generator, "run_quiz", MagicMock(...))`) —
  `run_quiz` (and its own collaborators `generate_questions`/
  `grade_answer`) already has its own tests; re-exercising it here would
  duplicate coverage instead of isolating the node's own orchestration
  logic.
- **`progress_coach_node`** mocks `get_coaching_message` and
  `try_study_buddy_assistance` directly, for the same reason — both
  already have dedicated tests. `progress_coach_node`'s call to
  `memory_set` is NOT mocked — it writes for real into the in-memory
  `memory_server._store` under the test's session ID, which is harmless
  (isolated per-session, nothing reads it back in this suite) but is a
  real, deliberate side effect worth knowing about.
- **`explainer_node`'s tool calls are not mocked.** `tool_list_files`,
  `tool_read_file`, etc. execute for real against the committed
  `study_materials/sample_notes/` files — a local disk read, no network,
  already treated as a real/fast/deterministic collaborator by
  `tests/test_filesystem_server.py`. Only the LLM boundary is mocked.

## Test cases

Coverage depth: happy path + key branches per node (not exhaustive) —
matches the depth already used elsewhere in this suite (e.g.
`tests/test_human_approval.py`).

### `curriculum_planner_node` — 1 test

The function is linear (no conditionals), so one test covers it fully:

- **Happy path:** mocked LLM returns a `StudyRoadmap`. Assert the
  returned dict has `roadmap` equal to that roadmap, `messages` equal to
  `[AIMessage]` with content `"Planned {n} topics over {w} weeks."`
  matching the roadmap's topic count and `total_weeks`, and `error` is
  `None`.

### `explainer_node` — 2 tests

- **Happy path:** `llm.invoke` is given a `side_effect` list of two
  responses — first a tool-call response (`tool_calls=[{"name":
  "tool_list_files", "args": {}, "id": "call_1"}]`), then a final
  response with `tool_calls=[]` and real explanation `content`. The real
  `tool_list_files` executes against `study_materials/sample_notes/`
  during the loop. Assert the node returns `{"messages":
  [final_response], "error": None}`, and assert `llm.invoke` was called
  exactly twice (proving exactly one real tool round-trip happened).
- **Branch — max iterations exceeded:** `llm.invoke`'s `return_value`
  always has non-empty `tool_calls` (never terminates the loop on its
  own). Assert the node returns `{"error": "explainer exceeded max
  iterations"}` (no `"messages"` key), and assert `llm.invoke` was
  called exactly `MAX_ITERATIONS` (8) times.

### `quiz_generator_node` — 2 tests

- **Happy path:** mocked `run_quiz` returns a `QuizResult(topic=...,
  score=0.8, passed=True, weak_areas=["recursion"])`. Seed
  `state["messages"]` with a final `AIMessage` so `get_last_explanation`
  has real content to extract. Assert `quiz_results` in the returned
  dict equals the prior `state["quiz_results"]` list plus the new
  result, `weak_areas` equals the correct merge, and `error` is `None`.
  Assert `run_quiz` was called with the current topic's title and the
  explanation text extracted from `state["messages"]`.
- **Branch — weak-area de-duplication:** seed `state["weak_areas"]` with
  an entry that also appears in the mocked `QuizResult.weak_areas`.
  Assert the returned `weak_areas` list contains no duplicate entries
  (compare as sorted lists, since the source uses `set()` and order
  isn't guaranteed).

### `progress_coach_node` — 2 tests

- **Happy path (passing score):** `state["quiz_results"]` ends with a
  `QuizResult(topic="X", score=0.9, passed=True, weak_areas=[])`;
  `state["roadmap"]` has at least one topic at
  `state["current_topic_index"]`. Mocked `get_coaching_message` returns
  a `CoachingMessage(summary="Great job!", tip="Keep going")`. Assert
  `roadmap.topics[idx].status == "completed"` (exercising the real,
  unmocked `next_topic_status`), `current_topic_index` is incremented by
  1, `messages == [AIMessage(content="Great job!")]`, `error is None`,
  and — using a `MagicMock` for `try_study_buddy_assistance` — assert it
  was **not** called (score is above `PASS_THRESHOLD`).
- **Branch (failing score triggers Study Buddy):** `QuizResult` with
  `score` below `PASS_THRESHOLD` and non-empty `weak_areas`. Assert
  `try_study_buddy_assistance` **was** called with the correct `topic`,
  `explanation` (from `get_last_explanation`), and `weak_areas`, and
  that the topic's status becomes `"needs_review"`.

Total: 7 new tests across 3 new files + 1 extended file.

## Explicitly out of scope

- Testing `progress_coach_node`'s defensive `idx < len(roadmap.topics)`
  guard before mutating topic status — this is a rare/defensive branch
  protected by the graph's own routing invariant
  (`session_is_complete`/`route_after_coach` already gate further calls
  once topics are exhausted), not a distinct product behavior worth a
  dedicated test at "key branches" depth.
- Asserting on `progress_coach_node`'s `print(...)` output when Study
  Buddy assistance is returned — cosmetic terminal output, not behavior.
- Any change to `human_approval_node` — it uses `langgraph.types.interrupt()`
  and requires the LangGraph runtime's interrupt machinery to test
  meaningfully; already excluded from this pass's scope (only
  `_parse_approval`/`route_after_approval`, its pure sub-functions, are
  tested today, and that's unchanged by this work).
- The Streamlit `@st.cache_resource` fix — separate, later piece of
  approved follow-up work, not part of this spec.
