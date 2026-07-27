import pytest

from learning_accelerator.mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
)


def test_list_study_files_sorted(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "b.md").write_text("second")
    (tmp_path / "a.md").write_text("first")

    assert list_study_files() == ["a.md", "b.md"]


def test_list_study_files_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path / "does-not-exist"))
    assert list_study_files() == []


def test_read_study_file_returns_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "note.md").write_text("hello world")

    assert read_study_file("note.md") == "hello world"


def test_read_study_file_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    assert read_study_file("missing.md") == "File not found: missing.md"


def test_read_study_file_blocks_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path.parent / "secret.md").write_text("top secret")

    with pytest.raises(ValueError, match="outside the notes directory"):
        read_study_file("../secret.md")


def test_search_notes_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "note.md").write_text("LangGraph is great\nOther line")

    assert search_notes("langgraph") == ["note.md: LangGraph is great"]


def test_search_notes_caps_at_20_results(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_PATH", str(tmp_path))
    (tmp_path / "note.md").write_text("\n".join(f"match {i}" for i in range(30)))

    assert len(search_notes("match")) == 20
