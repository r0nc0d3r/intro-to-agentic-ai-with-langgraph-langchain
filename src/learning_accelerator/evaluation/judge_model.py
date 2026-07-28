from __future__ import annotations

import os

from deepeval.models import DeepEvalBaseLLM

from learning_accelerator.config import get_chat_model


class LearningAcceleratorJudge(DeepEvalBaseLLM):
    """DeepEval judge model backed by this project's own provider-agnostic
    get_chat_model() — follows the same LLM_PROVIDER env var as the rest
    of the app (ollama/anthropic/openai), rather than being hardcoded to
    a single provider.
    """

    def load_model(self):
        return get_chat_model(temperature=0.0)

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        provider = os.environ.get("LLM_PROVIDER", "ollama")
        return f"learning-accelerator-judge/{provider}"


def get_judge_model() -> LearningAcceleratorJudge:
    return LearningAcceleratorJudge()
