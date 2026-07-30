from __future__ import annotations

from pathlib import Path

import pytest

from learning_accelerator.config import get_chat_model

NOTES_DIR = Path(__file__).parent.parent / "study_materials" / "sample_notes"


@pytest.fixture
def langgraph_basics_note_content() -> str:
    return (NOTES_DIR / "langgraph_basics.md").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_chat_model_cache():
    get_chat_model.cache_clear()
    yield
    get_chat_model.cache_clear()
