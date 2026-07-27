## Chapter 1: When to Use Multiple Agents

**Q: What's the default assumption — single agent or multi-agent?**
A: Single agent. Multi-agent adds coordination cost; only split when the
problem actually needs it.

**Q: When is a single agent enough?**
A: When there's one primary job that fits in one context window (e.g.
research-and-summarize, PR review, customer support, data extraction).

**Q: What are the 6 conditions that justify splitting into multiple agents?**
A: (1) distinct tool requirements, (2) divergent LLM call patterns
(single structured output vs. multi-turn tool loop), (3) different
temperature/model needs per task, (4) need for fault isolation, (5)
independent deployment needs, (6) cross-framework collaboration.

**Q: How many of those 6 conditions should typically be true before splitting?**
A: Two or more — no single condition alone usually justifies it.

**Q: What are the concrete costs of a multi-agent system?**
A: Shared state complexity (merge strategies for multiple writers),
harder debugging (failures surface after crossing agent boundaries),
latency multiplication (N agents = at least N LLM calls per run), and
more required infrastructure (checkpointing, observability, evaluation,
human oversight).

**Q: What's the gut-check heuristic for whether to split an agent boundary?**
A: If you can't explain why two tasks shouldn't be the same agent, they
probably shouldn't be split.

**Q: What are the Learning Accelerator's four agents?**
A: Curriculum Planner, Explainer, Quiz Generator, Progress Coach.

**Q: Why is the Curriculum Planner its own agent?**
A: Single deterministic call (`temperature=0.1`), structured JSON output,
no tools — mixing it with tool-calling agents would add noise.

**Q: Why is the Explainer its own agent?**
A: Multi-turn tool-calling loop (`temperature=0.3`) with a
non-deterministic iteration count — a different execution shape than a
single structured call.

**Q: Why is the Quiz Generator its own agent?**
A: Two LLM calls at different temperatures (0.4 generating, 0.1 grading),
plus it runs standalone over A2A — a dual-purpose pattern that doesn't
fit elsewhere.

**Q: Why is the Progress Coach its own agent?**
A: It makes the one cross-agent A2A call, synthesizes all other agents'
output, and owns the routing decision — coordination is its own concern.

**Q: Does this 4-agent split pattern generalize beyond learning/education?**
A: Yes — same shape (specialized agents + open protocols, only tool
access changes) appears in sales enablement, compliance training,
customer support, and engineering onboarding.
