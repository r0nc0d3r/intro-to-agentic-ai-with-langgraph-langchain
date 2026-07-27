## Chapter 3: Standardized Tool Access with MCP

**Q: What are the three primitives MCP defines?**
A: Tools (executable actions), Resources (read-only data by URI), and
Prompts (reusable prompt templates owned by the server).

**Q: Why does `memory_get` return the string `"null"` instead of Python
`None` for a missing key?**
A: To avoid `None`-handling edge cases when the result flows into LLM
tool output — a plain string is unambiguous everywhere it gets used.

**Q: How does `read_study_file` prevent path traversal?**
A: It resolves both the notes base directory and the requested file path
to absolute paths, then rejects the request unless the resolved file
path is actually inside the resolved base directory.

**Q: How does the Explainer agent talk to the MCP servers in this
chapter — via a real client/server connection, or something simpler?**
A: Something simpler (and this is deliberate, per the article): it
imports the server's plain Python functions directly and wraps each with
LangChain's `@tool` decorator, all in one process. A production setup
would swap in `MultiServerMCPClient` over a real subprocess transport —
the agent-facing tool-calling code doesn't change either way.

**Q: What temperature does the Explainer use, and why?**
A: `0.3` — balances multi-turn tool-calling reasoning with enough
consistency to stay on-task, unlike the Curriculum Planner's `0.1`.

**Q: What ends the Explainer's tool-calling loop?**
A: The LLM's response has no `tool_calls` (it gave a final explanation
instead of requesting another tool), or 8 iterations are reached,
whichever comes first.

**Q: Why must `ToolMessage.tool_call_id` match the id from the LLM's
`tool_calls` request?**
A: The LLM correlates each tool result back to the specific call it made
by that id — without a match, it can't tell which result answers which
request.

**Q: What does chapter 3's graph look like now?**
A: `START → curriculum_planner → explainer → END`. `human_approval`
still doesn't exist — it's inserted between them in chapter 5.

**Q: Which local Ollama model worked for the Explainer's tool-calling
loop, and were there any surprises?**
A: `gemma4:12b` correctly drives the whole loop — `tool_list_files` →
`tool_read_file` → `tool_memory_set` → `tool_read_file` → final answer —
and the final explanation is grounded in the actual note content it just
read. But it's not perfectly consistent run to run: one full end-to-end
run produced a fluent final explanation that didn't reflect the notes it
should have read at all (plausible completion from the model's prior
knowledge instead of the tool results), even though a repeat run at the
same temperature (0.3) grounded correctly. Lesson: passing tool-calling
mechanically doesn't guarantee the final synthesis actually uses the
retrieved content — that's a real reliability gap worth watching for,
not just a one-off fluke.
