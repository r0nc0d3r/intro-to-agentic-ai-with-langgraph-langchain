"""UI-layer tests for streamlit_app.py using Streamlit's AppTest harness.

These tests never touch a real LLM or LangGraph checkpoint: ``build_graph``
is replaced with a MagicMock so ``ui_graph.invoke`` / ``.update_state`` /
``.get_state`` return canned results, and the quiz agent functions
(``generate_questions`` / ``grade_answer``) are stubbed too. The goal is to
cover the Streamlit screen-routing and session-state logic itself — the
graph/agent behavior already has its own test coverage elsewhere.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from learning_accelerator.agents.quiz_generator import GradeResult, QuizQuestion
from learning_accelerator.graph.state import QuizResult, StudyRoadmap, Topic

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _mock_build_graph(monkeypatch, mock_graph: MagicMock) -> None:
    monkeypatch.setattr(
        "learning_accelerator.graph.workflow.build_graph",
        lambda **kwargs: mock_graph,
    )


def _roadmap(n_topics: int = 1, goal: str = "Learn X") -> StudyRoadmap:
    topics = [
        Topic(title=f"Topic {i}", description=f"desc {i}", estimated_minutes=30)
        for i in range(1, n_topics + 1)
    ]
    return StudyRoadmap(goal=goal, total_weeks=1, weekly_hours=5, topics=topics)


def _questions() -> list[QuizQuestion]:
    return [
        QuizQuestion(question=f"Q{i}?", expected_answer=f"A{i}", difficulty=d)
        for i, d in enumerate(("easy", "medium", "hard"), start=1)
    ]


def _click(at: AppTest, label_substring: str) -> None:
    matches = [b for b in at.button if label_substring in b.label]
    assert len(matches) == 1, f"expected exactly 1 button matching {label_substring!r}, got {[b.label for b in at.button]}"
    matches[0].click().run()


def _submit_goal(at: AppTest, goal: str) -> None:
    at.text_input[0].input(goal).run()
    _click(at, "Build Study Plan")


def _answer_current_question(at: AppTest, text: str = "my answer") -> None:
    at.text_area[0].input(text).run()
    _click(at, "Submit Answer")


# --- Session-state init & reset -------------------------------------------


def test_initial_screen_is_goal_input(monkeypatch):
    _mock_build_graph(monkeypatch, MagicMock())

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert at.session_state["screen"] == "GOAL_INPUT"
    assert at.title[0].value == "🎓 Learning Accelerator"


def test_empty_goal_shows_validation_error_and_does_not_transition(monkeypatch):
    _mock_build_graph(monkeypatch, MagicMock())

    at = AppTest.from_file(APP_PATH)
    at.run()
    _submit_goal(at, "   ")

    assert not at.exception
    assert at.session_state["screen"] == "GOAL_INPUT"
    assert any("learning goal" in e.value for e in at.error)


# --- Planning approval loop -------------------------------------------------


def test_submitting_goal_builds_roadmap_and_shows_approval_screen(monkeypatch):
    roadmap = _roadmap(2)
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "__interrupt__": [types.SimpleNamespace(value={"roadmap": roadmap})]
    }
    _mock_build_graph(monkeypatch, mock_graph)

    at = AppTest.from_file(APP_PATH)
    at.run()
    _submit_goal(at, "Learn LangGraph checkpointing")

    assert not at.exception
    assert at.session_state["screen"] == "ROADMAP_APPROVAL"
    assert at.title[0].value == "📋 Your Study Plan"
    assert any("Topic 1" in m.value for m in at.markdown)


def test_rejecting_roadmap_requests_a_new_plan_and_stays_on_approval_screen(monkeypatch):
    first_roadmap = _roadmap(1, goal="v1")
    second_roadmap = _roadmap(1, goal="v2")
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": first_roadmap})]},
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": second_roadmap})]},
    ]
    _mock_build_graph(monkeypatch, mock_graph)

    at = AppTest.from_file(APP_PATH)
    at.run()
    _submit_goal(at, "Learn X")
    _click(at, "No, generate a different plan")

    assert not at.exception
    assert at.session_state["screen"] == "ROADMAP_APPROVAL"
    assert at.session_state["roadmap"].goal == "v2"


def test_missing_roadmap_on_approval_screen_shows_error_and_allows_start_over(monkeypatch):
    _mock_build_graph(monkeypatch, MagicMock())

    at = AppTest.from_file(APP_PATH)
    at.session_state["screen"] = "ROADMAP_APPROVAL"
    at.session_state["roadmap"] = None
    at.run()

    assert any("No roadmap found" in e.value for e in at.error)

    _click(at, "Start over")

    assert not at.exception
    assert at.session_state["screen"] == "GOAL_INPUT"


# --- Quiz question/answer/grade cycle --------------------------------------


def _advance_to_quizzing(at: AppTest, monkeypatch, roadmap: StudyRoadmap, mock_graph: MagicMock) -> None:
    """Drive the app from GOAL_INPUT through to the first quiz question."""
    monkeypatch.setattr(
        "learning_accelerator.agents.quiz_generator.generate_questions",
        lambda *a, **k: _questions(),
    )
    at.run()
    _submit_goal(at, roadmap.goal)
    _click(at, "Yes, start studying")
    _click(at, "Start Quiz")


def test_quiz_shows_question_progress_for_current_question(monkeypatch):
    roadmap = _roadmap(1)
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": roadmap})]},
        {
            "messages": [AIMessage(content="explanation")],
            "roadmap": roadmap,
            "current_topic_index": 0,
        },
    ]
    _mock_build_graph(monkeypatch, mock_graph)

    at = AppTest.from_file(APP_PATH)
    _advance_to_quizzing(at, monkeypatch, roadmap, mock_graph)

    assert not at.exception
    assert at.session_state["screen"] == "QUIZZING"
    progress_texts = [p.proto.text for p in at.get("progress")]
    assert "Question 1 of 3" in progress_texts


def test_grading_appends_result_and_advances_to_next_question(monkeypatch):
    roadmap = _roadmap(1)
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": roadmap})]},
        {
            "messages": [AIMessage(content="explanation")],
            "roadmap": roadmap,
            "current_topic_index": 0,
        },
    ]
    _mock_build_graph(monkeypatch, mock_graph)
    monkeypatch.setattr(
        "learning_accelerator.agents.quiz_generator.grade_answer",
        lambda *a, **k: GradeResult(correct=True, score=1.0, feedback="Nice", missing_concept=""),
    )

    at = AppTest.from_file(APP_PATH)
    _advance_to_quizzing(at, monkeypatch, roadmap, mock_graph)
    _answer_current_question(at)

    assert not at.exception
    assert at.session_state["current_question_idx"] == 1
    assert len(at.session_state["graded_answers"]) == 1
    progress_texts = [p.proto.text for p in at.get("progress")]
    assert "Question 2 of 3" in progress_texts


def test_completing_last_question_hides_stale_question_progress_label(monkeypatch):
    """Regression test: the progress bar used to read "Question 4 of 3"
    for a beat after the last question of a 3-question quiz was graded.
    """
    roadmap = _roadmap(1)
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": roadmap})]},
        {
            "messages": [AIMessage(content="explanation")],
            "roadmap": roadmap,
            "current_topic_index": 0,
        },
    ]
    _mock_build_graph(monkeypatch, mock_graph)
    monkeypatch.setattr(
        "learning_accelerator.agents.quiz_generator.grade_answer",
        lambda *a, **k: GradeResult(correct=True, score=1.0, feedback="Nice", missing_concept=""),
    )

    at = AppTest.from_file(APP_PATH)
    _advance_to_quizzing(at, monkeypatch, roadmap, mock_graph)

    for _ in range(3):
        _answer_current_question(at)

    assert not at.exception
    assert at.session_state["current_question_idx"] == 3
    progress_texts = [p.proto.text for p in at.get("progress")]
    assert not any(t.startswith("Question") for t in progress_texts)
    assert any(s.value == "Quiz complete!" for s in at.success)
    assert any(m.value == "100%" for m in at.metric)


# --- Coaching / advance-to-next-topic + final results screen ---------------


def test_continue_after_quiz_advances_to_next_topic_with_coaching_message(monkeypatch):
    roadmap = _roadmap(2)
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": roadmap})]},
        {
            "messages": [AIMessage(content="explanation 1")],
            "roadmap": roadmap,
            "current_topic_index": 0,
        },
        {
            "messages": [AIMessage(content="Great job!"), AIMessage(content="explanation 2")],
            "quiz_results": [],
            "weak_areas": [],
            "current_topic_index": 1,
            "roadmap": roadmap,
            "error": None,
        },
    ]
    mock_graph.get_state.return_value = types.SimpleNamespace(values={"messages": []})
    _mock_build_graph(monkeypatch, mock_graph)
    monkeypatch.setattr(
        "learning_accelerator.agents.quiz_generator.grade_answer",
        lambda *a, **k: GradeResult(correct=True, score=1.0, feedback="Nice", missing_concept=""),
    )

    at = AppTest.from_file(APP_PATH)
    _advance_to_quizzing(at, monkeypatch, roadmap, mock_graph)
    for _ in range(3):
        _answer_current_question(at)
    _click(at, "Continue")

    assert not at.exception
    assert at.session_state["screen"] == "EXPLAINING"
    assert at.session_state["current_topic_index"] == 1
    assert at.session_state["topic_title"] == "Topic 2"
    assert at.session_state["coaching_message"] == "Great job!"
    assert any("Coach:" in i.value for i in at.info)


def test_session_complete_screen_shows_overall_average_and_per_topic_results(monkeypatch):
    roadmap = _roadmap(1)
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"__interrupt__": [types.SimpleNamespace(value={"roadmap": roadmap})]},
        {
            "messages": [AIMessage(content="explanation")],
            "roadmap": roadmap,
            "current_topic_index": 0,
        },
        {
            "messages": [AIMessage(content="Great job!")],
            "quiz_results": [QuizResult(topic="Topic 1", score=1.0, passed=True)],
            "weak_areas": [],
            "current_topic_index": 1,
            "roadmap": roadmap,
            "error": None,
        },
    ]
    mock_graph.get_state.return_value = types.SimpleNamespace(values={"messages": []})
    _mock_build_graph(monkeypatch, mock_graph)
    monkeypatch.setattr(
        "learning_accelerator.agents.quiz_generator.grade_answer",
        lambda *a, **k: GradeResult(correct=True, score=1.0, feedback="Nice", missing_concept=""),
    )

    at = AppTest.from_file(APP_PATH)
    _advance_to_quizzing(at, monkeypatch, roadmap, mock_graph)
    for _ in range(3):
        _answer_current_question(at)
    _click(at, "Continue")

    assert not at.exception
    assert at.session_state["screen"] == "COMPLETE"
    assert at.title[0].value == "🎉 Session Complete!"
    assert any(m.label == "Overall Average" and m.value == "100%" for m in at.metric)
    assert any("Topic 1" in m.value for m in at.markdown)

    _click(at, "Start a New Session")

    assert not at.exception
    assert at.session_state["screen"] == "GOAL_INPUT"
    assert at.session_state["session_id"] is None


# --- Error display -----------------------------------------------------------


def test_display_error_shows_message_and_start_over_resets_session(monkeypatch):
    _mock_build_graph(monkeypatch, MagicMock())

    at = AppTest.from_file(APP_PATH)
    at.session_state["screen"] = "EXPLAINING"
    at.session_state["error"] = "boom"
    at.session_state["roadmap"] = _roadmap(1)
    at.session_state["topic_title"] = "Topic 1"
    at.run()

    assert any("boom" in e.value for e in at.error)

    _click(at, "Start over")

    assert not at.exception
    assert at.session_state["screen"] == "GOAL_INPUT"
