# Streamlit `@st.cache_resource` Fix

## Goal

`streamlit_app.py` calls `build_graph(...)` at module scope to construct
`ui_graph`. Streamlit re-executes the entire script top-to-bottom on
every widget interaction (a "rerun"), so `ui_graph` — and the SQLite
connection / `StateGraph` compile inside it — gets rebuilt on every
click, not just once per session. Flagged as a Recommendation (not a
blocking finding, measured impact ~5ms per rebuild) by the final
whole-branch review of the runtime-performance pass (PR #19), which
suggested wrapping graph construction in `@st.cache_resource` so
Streamlit caches it across reruns.

This is the second of two approved follow-ups from that pass; the first
(mocked node-function tests) shipped as PR #20.

## The core fix

**File:** `streamlit_app.py`

Replace the current module-level eager construction:

```python
ui_graph = build_graph(
    db_path=".data/checkpoints_ui.sqlite",
    interrupt_before=["quiz_generator"],
)
```

with:

```python
@st.cache_resource
def get_ui_graph():
    return build_graph(
        db_path=".data/checkpoints_ui.sqlite",
        interrupt_before=["quiz_generator"],
    )


ui_graph = get_ui_graph()
```

- `get_ui_graph()` is called every rerun (the script still re-executes
  top-to-bottom, as it does today), but `@st.cache_resource` makes every
  call after the first a cheap cache lookup instead of a real
  `build_graph()` call — the SQLite connection and `StateGraph` compile
  happen once, not per click.
- Every existing `ui_graph.invoke(...)`, `.update_state(...)`,
  `.get_state(...)` call site elsewhere in the file is unchanged —
  `ui_graph` is still a plain module-level name bound to the graph
  object, just now assigned via a cached call instead of a direct one.
- `st.cache_resource`'s cache is process-wide, not per-browser-session
  (unlike `st.session_state`) — for this single-user local app, that's
  the correct behavior (one shared graph backing the one SQLite file),
  not a concern to guard against.

## Test isolation companion fix

**File:** `tests/test_streamlit_app.py`

**Verified empirically** (not assumed) before finalizing this design: a
throwaway probe (`@st.cache_resource`-decorated counter function, run
via `streamlit.testing.v1.AppTest.from_file()` across two separate
`AppTest` instances in the same pytest process) confirmed that
`st.cache_resource` state persists across separate `AppTest.from_file()`
calls — the second instance saw the first instance's cached value, not a
fresh one. Calling `st.cache_resource.clear()` before each `AppTest`
instance reliably resets it (confirmed in the same probe).

Without this fix, once `ui_graph` construction is cached, the first test
in `tests/test_streamlit_app.py` to run would populate the cache with
its `MagicMock`, and every later test would silently receive that same
stale mock instead of its own — breaking the whole file's test
isolation.

**Fix:** add one line to the existing shared helper (called by all 17
tests in the file today — confirmed via grep, no test bypasses it):

```python
def _mock_build_graph(monkeypatch, mock_graph: MagicMock) -> None:
    st.cache_resource.clear()
    monkeypatch.setattr(
        "learning_accelerator.graph.workflow.build_graph",
        lambda **kwargs: mock_graph,
    )
```

Requires adding `import streamlit as st` to the test file's imports (not
currently imported there).

No other test code changes. All 17 existing tests already route through
this helper.

## Testing

The existing `tests/test_streamlit_app.py` suite (17 tests, covering all
5 screens and the full session flow) is both the verification and the
regression guard: if cache isolation were broken, existing tests would
start failing or cross-contaminating immediately, since each test
asserts against its own distinct `MagicMock` return values. No new test
cases are needed beyond confirming the existing suite stays green after
both changes.

## Explicitly out of scope

- Any other Streamlit performance optimization — this fix is scoped
  exactly to the one issue the prior review flagged.
- Changes to `main.py`'s graph construction (already fixed via lazy
  `get_default_graph()` in the runtime-performance pass) or to
  `build_graph()` itself — unchanged by this work.
