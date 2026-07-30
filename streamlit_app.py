"""
streamlit_app.py

Streamlit web interface for the Learning Accelerator.

Runs the same LangGraph graph as main.py — only the I/O mechanism
changes. Instead of terminal input/output, this uses Streamlit widgets
and session state.

Run:
    uv run streamlit run streamlit_app.py

Architecture:
    Five screens: GOAL_INPUT -> ROADMAP_APPROVAL -> EXPLAINING ->
    QUIZZING -> COMPLETE.

    A separate graph instance (ui_graph) is compiled with
    interrupt_before=["quiz_generator"] so the graph pauses before the
    quiz step and returns control to Streamlit. The UI handles quiz I/O
    directly (calling generate_questions and grade_answer), then injects
    the QuizResult into the checkpoint via ui_graph.update_state() and
    resumes execution from progress_coach onward.

    This means: zero changes to quiz_generator_node or run_quiz(); the
    terminal interface (main.py) is completely unaffected; the LangGraph
    graph code is identical, only I/O changes.
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage
from langgraph.types import Command

from learning_accelerator.agents.quiz_generator import generate_questions, grade_answer
from learning_accelerator.graph.state import GradedAnswer, QuizResult, initial_state
from learning_accelerator.graph.workflow import build_graph
from learning_accelerator.observability.langfuse_setup import flush_langfuse, get_langfuse_config

# Separate checkpoint file from main.py's (.data/checkpoints.sqlite) — this
# graph is compiled with interrupt_before=["quiz_generator"], a different
# compiled graph than main.py's, so a session started in one interface
# can't currently be resumed in the other.
@st.cache_resource
def get_ui_graph():
    return build_graph(
        db_path=".data/checkpoints_ui.sqlite",
        interrupt_before=["quiz_generator"],
    )


ui_graph = get_ui_graph()

st.set_page_config(page_title="Learning Accelerator", page_icon="🎓", layout="centered")


def init_state() -> None:
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "roadmap": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "quiz_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "coaching_message": "",
        "error": None,
        "goal": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def go_to(screen: str) -> None:
    st.session_state.screen = screen


def extract_explanation(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""


def first_ai_message(messages: list) -> str:
    """Get the first AIMessage with content in a slice of newly-appended
    messages. Used for the coaching message: progress_coach always runs
    before explainer in the single invoke that follows a quiz, and both
    append a plain (no-tool-calls) AIMessage — taking the *first* one
    (progress_coach's) instead of the *last* (which would collide with
    extract_explanation and always resolve to explainer's message
    instead) is what actually distinguishes them.
    """
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def new_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def start_session(goal: str) -> None:
    """Runs: curriculum_planner -> human_approval (interrupt)."""
    session_id = str(uuid.uuid4())[:8]
    config = get_langfuse_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config
    st.session_state.goal = goal

    state = initial_state(goal, session_id)

    with st.spinner("Building your study roadmap..."):
        result = ui_graph.invoke(state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
    elif result.get("error"):
        st.session_state.error = result["error"]
    else:
        st.session_state.error = "Unexpected: no interrupt after planner."


def approve_roadmap(approved: bool) -> None:
    """If approved: human_approval -> explainer, then pauses before quiz_generator.
    If rejected: human_approval -> curriculum_planner -> interrupt again."""
    decision = "yes" if approved else "no"

    with st.spinner("Starting your study session..." if approved else "Generating a new plan..."):
        result = ui_graph.invoke(Command(resume=decision), config=st.session_state.graph_config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
        return

    if result.get("error"):
        st.session_state.error = result["error"]
        return

    messages = result.get("messages", [])
    st.session_state.explanation = extract_explanation(messages)

    roadmap = result.get("roadmap") or st.session_state.roadmap
    st.session_state.roadmap = roadmap
    idx = result.get("current_topic_index", 0)
    st.session_state.current_topic_index = idx

    topic = roadmap.topics[idx]
    st.session_state.topic_title = topic.title
    st.session_state.topic_description = topic.description

    with st.spinner("Generating quiz questions..."):
        questions = generate_questions(topic.title, st.session_state.explanation, n=3)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []

    go_to("EXPLAINING")


def advance_after_quiz(quiz_result: QuizResult) -> None:
    """Inject the QuizResult as if quiz_generator ran, then resume from
    progress_coach onward."""
    config = st.session_state.graph_config
    existing = st.session_state.quiz_results
    all_weak = list(set(st.session_state.weak_areas + quiz_result.weak_areas))

    ui_graph.update_state(
        config,
        {
            "quiz_results": existing + [quiz_result],
            "weak_areas": all_weak,
            "roadmap": st.session_state.roadmap,
            "current_topic_index": st.session_state.current_topic_index,
            "error": None,
        },
        as_node="quiz_generator",
    )

    prior_message_count = len(ui_graph.get_state(config).values.get("messages", []))

    with st.spinner("Getting coaching feedback..."):
        result = ui_graph.invoke(None, config=config)

    if result.get("error"):
        st.session_state.error = result["error"]
        return

    # This one invoke runs progress_coach (always) and then explainer (if
    # there's a next topic) with no interrupt boundary between them, so
    # both nodes' plain AIMessages land in the same result["messages"].
    # Scope to only what THIS invoke appended, so a later topic's
    # explanation can't bleed into the coaching message extraction below.
    messages = result.get("messages", [])
    new_messages = messages[prior_message_count:]
    st.session_state.coaching_message = first_ai_message(new_messages)
    st.session_state.quiz_results = result.get("quiz_results", existing + [quiz_result])
    st.session_state.weak_areas = result.get("weak_areas", all_weak)
    new_idx = result.get("current_topic_index", st.session_state.current_topic_index + 1)
    st.session_state.current_topic_index = new_idx

    roadmap = result.get("roadmap", st.session_state.roadmap)
    st.session_state.roadmap = roadmap

    if roadmap is None or new_idx >= len(roadmap.topics):
        flush_langfuse()
        go_to("COMPLETE")
        return

    st.session_state.explanation = extract_explanation(new_messages)
    topic = roadmap.topics[new_idx]
    st.session_state.topic_title = topic.title
    st.session_state.topic_description = topic.description

    with st.spinner("Generating quiz questions..."):
        questions = generate_questions(topic.title, st.session_state.explanation, n=3)

    st.session_state.quiz_questions = questions
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []

    go_to("EXPLAINING")


def screen_goal_input() -> None:
    st.title("🎓 Learning Accelerator")
    st.markdown(
        "Enter a learning goal and the system will build a personalised "
        "study plan, explain each topic using your notes, and quiz you "
        "as you go, all running locally with Ollama."
    )

    with st.form("goal_form"):
        goal = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. Learn LangGraph checkpointing from scratch",
        )
        submitted = st.form_submit_button("Build Study Plan →", type="primary")

    if submitted:
        if not goal.strip():
            st.error("Please enter a learning goal.")
        else:
            start_session(goal.strip())
            st.rerun()

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Try again"):
            st.session_state.error = None
            st.rerun()


def screen_roadmap_approval() -> None:
    st.title("📋 Your Study Plan")
    roadmap = st.session_state.roadmap

    if roadmap is None:
        st.error("No roadmap found.")
        if st.button("Start over"):
            new_session()
            st.rerun()
        return

    st.markdown(f"**Goal:** {roadmap.goal}")
    st.markdown(f"**Duration:** {roadmap.total_weeks} weeks @ {roadmap.weekly_hours} hrs/week")
    st.markdown("---")

    for i, topic in enumerate(roadmap.topics, 1):
        prereq_text = f" *(needs: {', '.join(topic.prerequisites)})*" if topic.prerequisites else ""
        st.markdown(f"**{i}. {topic.title}**, {topic.estimated_minutes} min{prereq_text}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{topic.description}")

    st.markdown("---")
    st.markdown("Does this study plan look good?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, start studying", type="primary", use_container_width=True):
            approve_roadmap(True)
            st.rerun()
    with col2:
        if st.button("🔄 No, generate a different plan", use_container_width=True):
            approve_roadmap(False)
            st.rerun()


def screen_explaining() -> None:
    roadmap = st.session_state.roadmap
    total = len(roadmap.topics) if roadmap else 1
    idx = st.session_state.current_topic_index

    st.progress(idx / total, text=f"Topic {idx + 1} of {total}")
    st.title(f"📖 {st.session_state.topic_title}")
    st.caption(st.session_state.topic_description)
    st.markdown("---")

    if st.session_state.coaching_message:
        st.info(f"💬 **Coach:** {st.session_state.coaching_message}")
        st.markdown("---")

    if st.session_state.explanation:
        st.markdown("### Explanation")
        st.markdown(st.session_state.explanation)
    else:
        st.warning("No explanation available, starting quiz with topic context.")

    st.markdown("---")
    st.markdown(f"**Ready to test your knowledge of *{st.session_state.topic_title}*?**")

    if st.button("Start Quiz →", type="primary"):
        st.session_state.coaching_message = ""
        go_to("QUIZZING")
        st.rerun()


def screen_quizzing() -> None:
    questions = st.session_state.quiz_questions
    q_idx = st.session_state.current_question_idx
    total_q = len(questions)
    roadmap = st.session_state.roadmap
    total_topics = len(roadmap.topics) if roadmap else 1
    topic_idx = st.session_state.current_topic_index

    st.progress(topic_idx / total_topics, text=f"Topic {topic_idx + 1} of {total_topics}")
    if total_q > 0 and q_idx < total_q:
        st.progress(q_idx / total_q, text=f"Question {q_idx + 1} of {total_q}")

    st.title(f"🧠 Quiz: {st.session_state.topic_title}")
    st.markdown("---")

    for i, graded in enumerate(st.session_state.graded_answers):
        status = "✅" if graded.correct else "❌"
        with st.expander(f"{status} Q{i + 1}: {graded.question[:80]}...", expanded=False):
            st.markdown(f"**Your answer:** {graded.user_answer}")
            st.markdown(f"**Score:** {graded.score:.0%}")
            st.markdown(f"**Feedback:** {graded.feedback}")

    if q_idx < total_q:
        q = questions[q_idx]

        st.markdown(f"**Question {q_idx + 1} [{q.difficulty}]:**")
        st.markdown(q.question)

        with st.form(f"answer_form_{q_idx}"):
            answer = st.text_area(
                "Your answer:", placeholder="Type your answer here...",
                height=120, key=f"answer_input_{q_idx}",
            )
            submitted = st.form_submit_button("Submit Answer →", type="primary")

        if submitted:
            user_answer = answer.strip() or "(no answer provided)"

            with st.spinner("Grading your answer..."):
                grade = grade_answer(q.question, q.expected_answer, user_answer)

            graded_answer = GradedAnswer(
                question=q.question,
                expected_answer=q.expected_answer,
                user_answer=user_answer,
                correct=grade.correct,
                feedback=grade.feedback,
                score=grade.score,
            )
            st.session_state.graded_answers.append(graded_answer)
            if grade.missing_concept:
                st.session_state.current_quiz_missing_concepts.append(grade.missing_concept)
            st.session_state.current_question_idx = q_idx + 1
            st.rerun()

    else:
        st.markdown("---")
        graded = st.session_state.graded_answers
        avg_score = sum(a.score for a in graded) / len(graded) if graded else 0.0
        weak_areas = list(dict.fromkeys(st.session_state.current_quiz_missing_concepts))

        st.success("✅ Quiz complete!")
        st.metric("Your score", f"{avg_score:.0%}")

        quiz_result = QuizResult(
            topic=st.session_state.topic_title,
            score=avg_score,
            passed=avg_score >= 0.5,
            weak_areas=weak_areas,
            questions=graded,
        )

        if st.button("Continue →", type="primary"):
            advance_after_quiz(quiz_result)
            st.rerun()


def screen_complete() -> None:
    st.title("🎉 Session Complete!")
    st.markdown("---")

    roadmap = st.session_state.roadmap
    quiz_results = st.session_state.quiz_results

    if roadmap:
        st.markdown(f"**Goal:** {roadmap.goal}")

    if quiz_results:
        avg = sum(r.score for r in quiz_results) / len(quiz_results)
        st.metric("Overall Average", f"{avg:.0%}")
        st.markdown("---")
        st.markdown("### Results by Topic")
        for r in quiz_results:
            status = "✅" if r.score >= 0.5 else "❌"
            weak = f", review: {', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
            st.markdown(f"{status} **{r.topic}**: {r.score:.0%}{weak}")

    if st.session_state.weak_areas:
        st.markdown("---")
        st.markdown("### Topics to Revisit")
        for w in st.session_state.weak_areas[:5]:
            st.markdown(f"- {w}")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Start a New Session", type="primary"):
        new_session()
        st.rerun()


def display_error() -> None:
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")
        if st.button("← Start over"):
            new_session()
            st.rerun()


screen = st.session_state.screen

if screen == "GOAL_INPUT":
    screen_goal_input()
elif screen == "ROADMAP_APPROVAL":
    display_error()
    screen_roadmap_approval()
elif screen == "EXPLAINING":
    display_error()
    screen_explaining()
elif screen == "QUIZZING":
    display_error()
    screen_quizzing()
elif screen == "COMPLETE":
    screen_complete()
else:
    st.error(f"Unknown screen: {screen}")
    if st.button("Reset"):
        new_session()
        st.rerun()
