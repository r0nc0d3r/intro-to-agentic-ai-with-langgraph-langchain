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
independent attempts, with two apparent causes: once the Explainer agent
itself sent a malformed tool-call argument (`{'1': ...}` instead of
`{'session_id': ...}`), and twice the judge model's own internal
`FaithfulnessMetric._a_generate_claims` sub-call returned empty/unparsable
JSON that DeepEval couldn't recover from. At the time, the second cause
was attributed to a `gemma4:12b` reliability limit.

**Q: Was that conclusion actually right?**
A: No — a final whole-branch review caught a real bug in our own code
that made the "local-model limitation" story incomplete. Our
`LearningAcceleratorJudge` wrapper (`judge_model.py`) defined
`generate(self, prompt: str)` / `a_generate(self, prompt: str)` with no
`schema` parameter. DeepEval's `DeepEvalBaseLLM.a_generate_with_schema()`
always tries `self.a_generate(*args, schema=schema, **kwargs)` first —
and when that raises `TypeError` (because our signature didn't accept
`schema`), it silently catches it and falls back to unstructured
`a_generate(*args, **kwargs)`. So every structured-output sub-call
DeepEval made through our judge — including `FaithfulnessMetric`'s
claim-extraction step — was silently downgraded to free-text generation
with no schema enforcement at all, even though the rest of this app
(chapters 2, 4) uses `with_structured_output()` everywhere. That is a far
more likely explanation for "empty/unparsable JSON" than the model simply
being unreliable.

**Q: What was the fix, and did it actually resolve the failure?**
A: Added `schema=None` to both `generate()` and `a_generate()`; when a
schema is passed, route through `model.with_structured_output(schema).invoke(prompt)`
instead of the plain `.invoke(prompt).content` path — matching the
pattern already used elsewhere in this codebase. Also switched to reusing
`self.model` (already built once in `DeepEvalBaseLLM.__init__`) instead of
calling `load_model()` fresh on every `generate()` call. After this fix, a
live re-run of `test_explanation_is_faithful_to_notes` against
`gemma4:12b` **passed, with a perfect 1.000 Faithfulness score** — strong
evidence the schema-dropping bug, not an inherent `gemma4:12b` limitation,
was the real root cause of at least the "unparsable JSON" failures.

**Q: Was the failing test's threshold lowered, or its wording changed, to
make it pass?**
A: No — neither the original investigation nor the eventual fix touched
the threshold or test wording. The first pass correctly ruled out
threshold/wording as the fix, but attributed the residual failures to the
wrong root cause; the follow-up review found and fixed the actual bug in
our own judge wrapper, and that's what turned the failures into a pass.

**Q: What does this whole episode actually teach about using a local 12B
model for LLM-as-judge evaluation?**
A: Two lessons, and it matters which one gets the credit. First, on
DeepEval's internals specifically: reading the installed
`deepeval==4.1.4` source shows every metric used here is multi-call, not
just `FaithfulnessMetric` — `FaithfulnessMetric` chains 4 sub-calls
(`_a_generate_truths`, `_a_generate_claims`, `_a_generate_verdicts`,
`_a_generate_reason`); `AnswerRelevancyMetric` chains 3
(`_a_generate_statements`, `_a_generate_verdicts`, `_a_generate_reason`);
`GEval` makes at least 2 when used with `criteria=` (as our tests do)
— steps generation, then evaluation. `FaithfulnessMetric` chains the
*most* sub-calls and feeds the *largest* prompt (the full retrieval
context) into that pipeline, giving it the widest surface for a malformed
completion — a real factor. But the second, bigger lesson is a humbling
one: don't blame the model before auditing your own integration code.
The judge wrapper silently dropping DeepEval's `schema` parameter meant
none of our structured-output sub-calls were ever actually schema-enforced
— a bug entirely in our code, invisible until someone read
`DeepEvalBaseLLM`'s source and noticed the `except TypeError: pass`
fallback. "The local model is unreliable" was a much more comfortable
conclusion than "our wrapper has been silently degrading every
structured call," and it was wrong.

**Q: Is there still a deeper limitation in using `gemma4:12b` as *both*
the agent-under-test and the judge model?**
A: Yes, and it's still worth naming honestly, even though the schema fix
resolved the observed failure: best practice is to judge with a model at
least as capable as (ideally stronger than) the model being evaluated, to
avoid the model grading its own homework. This repo's local setup only
has `gemma4:12b`-class models available, so the agent and judge here are
effectively the same capability tier. One passing re-run with a perfect
score doesn't retire this concern — it's a standing constraint of
local-only evaluation, not something any one bug fix resolves.

**Q: Where did chapter 7's actual source content come from, given the
article's own page didn't yield the chapter 7 text?**
A: The article references a companion reference-implementation repo
(`sandeepmb/freecodecamp-multi-agent-ai-system`). Its `tests/test_eval.py`
was pulled directly and adapted: our own Pydantic-model attribute access
instead of its dict access, our own "LangGraph Basics" domain content
instead of its "Python closures" example, and our own provider-agnostic
judge instead of its Ollama-only one.
