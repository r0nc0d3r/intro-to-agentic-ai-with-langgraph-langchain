from learning_accelerator.evaluation.judge_model import (
    LearningAcceleratorJudge,
    get_judge_model,
)


def test_get_judge_model_returns_judge_instance():
    judge = get_judge_model()
    assert isinstance(judge, LearningAcceleratorJudge)


def test_judge_model_name_includes_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    judge = get_judge_model()
    assert "ollama" in judge.get_model_name()


def test_judge_load_model_uses_configured_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:12b")
    judge = get_judge_model()
    model = judge.load_model()
    assert type(model).__name__ == "ChatOllama"
    assert model.temperature == 0.0
