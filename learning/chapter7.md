## Chapter 7: Evaluating Agent Quality with DeepEval

**Q: What DeepEval metrics were used, and what does each check?**
A: `FaithfulnessMetric` (does the Explainer's explanation avoid
hallucinating facts not in the source notes?), `AnswerRelevancyMetric`
(does the explanation actually address the question asked?), and
`GEval` (a custom-criteria judge, used for two things no pre-built
metric covers: whether quiz questions test genuine understanding vs.
rote recall, and whether coaching messages are specific/actionable
rather than generic).

**Q: Why does this chapter need a custom judge model wrapper instead of
just passing a model name string to DeepEval?**
A: `LearningAcceleratorJudge(DeepEvalBaseLLM)` reuses this repo's own
`config.get_chat_model()` — so the judge follows the same `LLM_PROVIDER`
env var (ollama/anthropic/openai) as every other agent, instead of being
hardcoded to one provider like the source reference's Ollama-only judge.

**Q: Why are all of `tests/test_eval.py`'s tests marked `@pytest.mark.eval`,
and why does that matter?**
A: Every test here calls a real agent function that makes live LLM calls
(`explainer_node`, `generate_questions`, `grade_answer`,
`get_coaching_message`) plus the judge model's own calls on top —
slow (30-120s+ each) and non-deterministic. `pyproject.toml` sets
`addopts = "-m 'not eval'"`, so plain `uv run pytest -v` still runs only
the fast 48 tests with zero network calls; `uv run pytest tests/test_eval.py -v -s -m eval`
opts in explicitly.

**Q: What were the real results running this against `gemma4:12b`?**
A: 11 of 12 tests passed, several with perfect 1.000 scores well above
the conservative 0.6 threshold. One test —
`test_explanation_is_faithful_to_notes` — failed 3 times out of 3
independent attempts, with two distinct root causes: once the Explainer
agent itself sent a malformed tool-call argument (`{'1': ...}` instead of
`{'session_id': ...}`), and twice the judge model's own internal
`FaithfulnessMetric._a_generate_claims` sub-call returned empty/unparsable
JSON that DeepEval couldn't recover from.

**Q: Was the failing test's threshold lowered, or its wording changed, to
make it pass?**
A: No — deliberately. The investigation traced both failure modes to
concrete causes (a tool-call bug and an empty LLM completion), neither of
which a threshold or wording change would fix. The test is left failing
and documented in-code as a known limitation, not quietly patched around.

**Q: What does this failing test actually teach about using a local 12B
model for LLM-as-judge evaluation?**
A: `FaithfulnessMetric` is DeepEval's most LLM-call-heavy metric — it
makes at least two separate structured-output sub-calls (truth
extraction, then claim extraction) to the judge. `AnswerRelevancyMetric`
and the `GEval` metrics call the same judge model but with a single,
simpler one-shot prompt, and passed cleanly. The lesson: a local model
that's fine for simple one-shot judging can still fail on a metric that
chains multiple structured-output calls together — more LLM calls in a
metric's internal pipeline means more chances for one of them to return
malformed output.

**Q: Is there a deeper limitation in using `gemma4:12b` as *both* the
agent-under-test and the judge model?**
A: Yes, and it's worth naming honestly: best practice is to judge with a
model at least as capable as (ideally stronger than) the model being
evaluated, to avoid the model grading its own homework. This repo's local
setup only has `gemma4:12b`-class models available, so the agent and
judge here are effectively the same capability tier — a real constraint
of local-only evaluation, not something this chapter's code can fix.

**Q: Where did chapter 7's actual source content come from, given the
article's own page didn't yield the chapter 7 text?**
A: The article references a companion reference-implementation repo
(`sandeepmb/freecodecamp-multi-agent-ai-system`). Its `tests/test_eval.py`
was pulled directly and adapted: our own Pydantic-model attribute access
instead of its dict access, our own "LangGraph Basics" domain content
instead of its "Python closures" example, and our own provider-agnostic
judge instead of its Ollama-only one.
