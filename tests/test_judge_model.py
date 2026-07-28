import asyncio
from unittest.mock import MagicMock

from pydantic import BaseModel

from learning_accelerator.evaluation.judge_model import (
    LearningAcceleratorJudge,
    get_judge_model,
)


class _TrivialSchema(BaseModel):
    answer: str


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


def test_generate_without_schema_calls_plain_invoke():
    """No schema passed -> behaves like a plain chat call, returning .content."""
    judge = get_judge_model()
    fake_message = MagicMock()
    fake_message.content = "plain text response"
    judge.model = MagicMock()
    judge.model.invoke.return_value = fake_message

    result = judge.generate("some prompt")

    judge.model.invoke.assert_called_once_with("some prompt")
    assert result == "plain text response"


def test_generate_with_schema_uses_structured_output():
    """A schema passed -> routed through with_structured_output(), matching
    how the rest of this codebase (chapters 2, 4) does structured output.
    The DeepEvalBaseLLM.generate_with_schema() wrapper calls
    generate(*args, schema=schema, **kwargs) first, so accepting this kwarg
    is what makes DeepEval's structured internal sub-calls (e.g.
    FaithfulnessMetric's claim extraction) actually schema-enforced instead
    of silently falling back to unstructured text.
    """
    judge = get_judge_model()
    structured_model = MagicMock()
    structured_model.invoke.return_value = _TrivialSchema(answer="42")
    judge.model = MagicMock()
    judge.model.with_structured_output.return_value = structured_model

    result = judge.generate("some prompt", schema=_TrivialSchema)

    judge.model.with_structured_output.assert_called_once_with(_TrivialSchema)
    structured_model.invoke.assert_called_once_with("some prompt")
    assert isinstance(result, _TrivialSchema)
    assert result.answer == "42"


def test_generate_reuses_self_model_instead_of_rebuilding(monkeypatch):
    """generate()/a_generate() should reuse self.model (built once in
    DeepEvalBaseLLM.__init__ via load_model()) rather than calling
    load_model() again on every generate call."""
    judge = get_judge_model()
    fake_model = MagicMock()
    fake_model.invoke.return_value = MagicMock(content="x")
    judge.model = fake_model
    call_count = {"n": 0}

    def fake_load_model():
        call_count["n"] += 1
        return fake_model

    monkeypatch.setattr(judge, "load_model", fake_load_model)

    judge.generate("prompt one")
    judge.generate("prompt two")

    assert call_count["n"] == 0


def test_a_generate_supports_schema_kwarg():
    judge = get_judge_model()
    structured_model = MagicMock()
    structured_model.invoke.return_value = _TrivialSchema(answer="async-42")
    judge.model = MagicMock()
    judge.model.with_structured_output.return_value = structured_model

    result = asyncio.run(judge.a_generate("some prompt", schema=_TrivialSchema))

    judge.model.with_structured_output.assert_called_once_with(_TrivialSchema)
    assert isinstance(result, _TrivialSchema)
    assert result.answer == "async-42"


def test_a_generate_without_schema_returns_plain_text():
    judge = get_judge_model()
    fake_message = MagicMock()
    fake_message.content = "async plain text"
    judge.model = MagicMock()
    judge.model.invoke.return_value = fake_message

    result = asyncio.run(judge.a_generate("some prompt"))

    assert result == "async plain text"


def test_get_model_name_lowercases_provider_and_includes_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:12b")
    judge = get_judge_model()
    name = judge.get_model_name()
    assert name == "learning-accelerator-judge/ollama/gemma4:12b"
