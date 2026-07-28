## Chapter 8: Cross-Framework Coordination with A2A

**Q: What problem does A2A actually solve?**
A: Framework isolation. Without a shared protocol, a LangGraph agent can't
ask a CrewAI agent to do work without coupling to CrewAI's internals (and
vice versa). A2A makes the *protocol* the integration point instead of a
framework-specific adapter — either side only needs to speak JSON-RPC
over HTTP, discoverable via a well-known Agent Card URL.

**Q: What is an Agent Card, and where is it served?**
A: A "business card" describing an A2A service's name, version, and
skills — served automatically at `/.well-known/agent-card.json`. Any
caller fetches this first to discover what a service can do before
sending it any real work.

**Q: Why does `quiz_service.py` convert `QuizQuestion`/`GradeResult`
(Pydantic models) to plain dicts before responding?**
A: Those Pydantic models exist for in-process type safety (chapter 4).
Once the data crosses an HTTP/JSON boundary, a dict is the natural
representation — there's no type safety to preserve across a wire
format, and the reference implementation's own dict-based design is
right for this specific boundary.

**Q: Does `try_a2a_quiz_delegation` actually get called anywhere in this
codebase's graph?**
A: No — it's defined and fully tested (mocked), matching what the
reference repo itself does, but nothing in `progress_coach_node` invokes
it. Documented honestly here rather than guessed at: it may be an
intentional extension point, or simply unfinished in the source
material. `try_study_buddy_assistance` IS wired in, for the low-score path.

**Q: What's the actual proof that this is genuine cross-framework
coordination, not just two services that happen to look similar?**
A: The CrewAI Study Buddy runs a completely different framework
(`crewai.Agent`/`Crew`/`Task`, not a LangGraph node) behind the identical
A2A protocol boundary as the Quiz Service. The Progress Coach calls it
through `request_study_assistance()` without knowing or caring that
CrewAI is on the other end — proven for real in this chapter's demo,
where a real `crew.kickoff()` executed against local Ollama and returned
genuine assistance text through the same `send_task()` client used for
the LangGraph-based Quiz Service.

**Q: What did fully-mocked unit tests miss, and why?**
A: Everything about the actual wire protocol. `tests/test_a2a_client.py`
mocks `httpx.get`/`httpx.post` entirely, so it never noticed that the
code was talking to the wrong endpoint. Only the chapter's demo script —
two real HTTP servers, real requests — exercised the genuine protocol
and immediately failed with a 404.

**Q: What protocol bugs did the real run actually find?**
A: Four, all in this codebase's A2A plumbing, none in CrewAI/Ollama/
litellm: (1) `send_task` posted to a nonexistent `/tasks/send` using the
legacy JSON-RPC method `tasks/send` — the installed `a2a-sdk` (0.3.26)
actually serves one endpoint at `/` with method `message/send`; (2) the
request/response `Part` schema uses a `kind` discriminator, not `type`,
and `Message` requires a `messageId` this codebase wasn't setting; (3)
both A2A executors read a nonexistent `context.current_request` attribute
— the real accessor is `context.get_user_input()` (and even if the
attribute had existed, the original `isinstance(part, TextPart)` check
would still have silently matched nothing, since `Part` is a Pydantic
`RootModel` wrapper, not `TextPart` itself); (4) outgoing `Message()`
construction omitted the required `message_id`.

**Q: What's the lesson from finding those bugs specifically via the demo
script, not via pytest?**
A: This is the clearest example yet in this course of why mocked unit
tests and a real end-to-end run are *both* necessary, not redundant.
Mocking `httpx` verifies your own call-site logic in isolation; it can
never catch "the remote API doesn't actually work the way you assumed,"
because the mock encodes exactly that same wrong assumption. Only a real
request against the real installed dependency surfaces a genuine protocol
mismatch.

**Q: Was `a2a-sdk` pinned to an exact version, and why does that matter here?**
A: Yes — `a2a-sdk==0.3.26`, discovered during this chapter's first task.
Latest `a2a-sdk` (1.x) made a breaking protobuf-schema rewrite that drops
`A2AStarletteApplication` and several types this chapter depends on
entirely. Pinning to the last pre-break release keeps this chapter's code
working; a future upgrade would need real migration work, not just a
version bump.

**Q: Where does `try_study_buddy_assistance`'s `explanation` argument come
from in `progress_coach_node` — and why not just pass an empty string?**
A: `progress_coach_node` walks `state["messages"]` in reverse and takes the
last `AIMessage` with content and no `tool_calls` as the explanation, the
same pattern `quiz_generator_node` already uses to recover the Explainer's
output. The plan's alternative — passing `""` — was considered but rejected:
an empty explanation would give the Study Buddy nothing concrete to riff on,
producing a generic "fresh analogy" untethered to what the student actually
read, instead of supplementary help that responds to the real content.

**Q: What real result did the end-to-end demo produce?**
A: Both services came up (Agent Cards discoverable at ports 9001/9002),
the Quiz Service returned `"questions_ready"` with 3 real generated
questions, and the CrewAI Study Buddy returned `"complete"` with 1211
characters of genuine assistance text — a fresh analogy ("checkpointing
is like a video game save point"), a concrete example tying `thread_id`
and `SqliteSaver` to the analogy, and a memory tip — all produced by a
real `crew.kickoff()` against local Ollama (`gemma4:12b`, confirmed
working for both this repo's own LangChain calls and CrewAI's `LLM`
wrapper, with no special model-string workaround needed).
