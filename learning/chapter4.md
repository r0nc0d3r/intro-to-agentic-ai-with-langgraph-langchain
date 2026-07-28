## Chapter 4: Building the Four-Agent System

**Q: Why does the Quiz Generator use two different LLM calls instead of one?**
A: Generating questions wants creative variety (`temperature=0.4`);
grading answers wants consistent, analytical scoring (`temperature=0.1`).
One shared temperature would compromise both.

**Q: What determines whether a topic is marked "completed" or
"needs_review"?**
A: `next_topic_status(score)` — `"completed"` if the quiz average score
is `>= PASS_THRESHOLD` (0.5), otherwise `"needs_review"`.

**Q: When does `current_topic_index` advance — before or after the topic
status update?**
A: After. `progress_coach_node` sets `roadmap.topics[idx].status` first,
then returns `current_topic_index: idx + 1`.

**Q: How does the graph know when to stop looping through topics?**
A: `route_after_coach` calls `session_is_complete(state)`, which checks
`current_topic_index >= len(roadmap.topics)`. True routes to `END`;
false routes back to `"explainer"` for the next topic.

**Q: Why does `run_quiz` take an injectable `answer_source` parameter
instead of always calling `input()` directly like the article?**
A: The article's version blocks on real interactive stdin, which can't be
driven from this session's tooling. `answer_source` defaults to real
`input` for genuine interactive use, but the chapter 4 demo swaps in a
canned answer function so the full graph can run end-to-end without a
live terminal.

**Q: Why doesn't chapter 4's graph include `human_approval` yet, even
though the source article's chapter 4 code does?**
A: Our own spec deliberately assigns `human_approval` + `interrupt()` to
chapter 5. This chapter keeps the direct `curriculum_planner → explainer`
edge from chapter 3; chapter 5 inserts `human_approval` between them.

**Q: What does chapter 4's graph look like now?**
A: `START → curriculum_planner → explainer → quiz_generator →
progress_coach →(conditional)→ explainer | END`.

**Q: Which local Ollama model was used to verify the full end-to-end loop,
and what happened?**
A: `gemma4:12b` — same model that worked for chapters 2-3. A full run
correctly looped through all 5 roadmap topics (`current_topic_index`
0→5, then routed to `END`), generating and grading 3 questions per topic
via `with_structured_output` the whole way. Every topic landed on
`needs_review` because the canned demo answer ("I'm not fully sure...")
genuinely contains no content — the grader correctly scored it near
zero. That's the grading logic working as intended, not a bug: a real
learner's actual answers would score accordingly.
