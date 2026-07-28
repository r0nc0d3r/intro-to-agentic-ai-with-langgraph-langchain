from __future__ import annotations

from pathlib import Path

import pytest

NOTES_DIR = Path(__file__).parent.parent / "study_materials" / "sample_notes"


@pytest.fixture
def langgraph_basics_note_content() -> str:
    return (NOTES_DIR / "langgraph_basics.md").read_text(encoding="utf-8")
