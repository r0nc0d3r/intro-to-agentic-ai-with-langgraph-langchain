from learning_accelerator.graph.state import QuizResult, StudyRoadmap, Topic


def test_print_session_summary_shows_average_and_topics(capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import print_session_summary

    roadmap = StudyRoadmap(
        goal="Learn LangGraph",
        total_weeks=2,
        topics=[
            Topic(title="Nodes and Edges", description="d", estimated_minutes=30),
            Topic(title="Checkpointing", description="d", estimated_minutes=30),
        ],
    )
    result = {
        "roadmap": roadmap,
        "quiz_results": [
            QuizResult(topic="Nodes and Edges", score=1.0, passed=True),
            QuizResult(topic="Checkpointing", score=0.4, passed=False, weak_areas=["thread_id"]),
        ],
        "weak_areas": ["thread_id"],
    }

    print_session_summary(result)

    captured = capsys.readouterr()
    assert "Learn LangGraph" in captured.out
    assert "Nodes and Edges" in captured.out
    assert "Checkpointing" in captured.out
    assert "70%" in captured.out  # average of 1.0 and 0.4
    assert "thread_id" in captured.out


def test_print_session_summary_handles_no_roadmap(capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import print_session_summary

    print_session_summary({"roadmap": None})

    captured = capsys.readouterr()
    assert captured.out == ""
