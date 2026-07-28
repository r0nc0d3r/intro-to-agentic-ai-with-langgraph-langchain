## Chapter 6: Observability with Langfuse

**Q: How does Langfuse hook into a LangGraph run?**
A: Via LangChain's callback mechanism — `CallbackHandler` from
`langfuse.langchain` gets passed through `config["callbacks"]` at
`graph.invoke()` time, same as any other LangChain callback.

**Q: Why don't any of the existing agent node functions need to change
to get their LLM calls traced?**
A: LangGraph propagates the invoke-time `config` (including `callbacks`)
into every node's execution via Python contextvars, and LangChain's
`ChatModel.invoke()` reads that ambient config automatically through
`ensure_config()` when no explicit config is passed. Tracing is
transparent to node code.

**Q: How does a Langfuse trace get linked to our session/thread?**
A: Via the `metadata={"langfuse_session_id": session_id}` key in the
invoke config — `CallbackHandler` reads that specific metadata key and
maps it to Langfuse's own `session_id` trace attribute.

**Q: What happens if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't set?**
A: `get_langfuse_config()` omits the `callbacks` key entirely — the graph
runs exactly as it did in chapters 2-5, with zero tracing overhead and
no error. Tracing is opt-in, not required.

**Q: Why didn't this chapter stand up a real self-hosted Langfuse instance
to verify tracing?**
A: Langfuse's v3 self-host stack needs postgres + clickhouse + redis +
minio + two Langfuse services — too heavy to spin up just to check a
callback wiring. `docker-compose.langfuse.yml` is included as a reference
for anyone who wants to run one for real; verification instead used a
tiny local HTTP server standing in for the ingestion endpoint.

**Q: What did the mock-endpoint demo actually prove, and what does it NOT prove?**
A: It proves our integration code genuinely fires a real HTTP POST at
whatever `LANGFUSE_HOST` is configured to — the thing our code is
responsible for (confirmed: a 22.5KB request body, consistent with a
real trace payload, not a stray ping). It does NOT prove Langfuse's own
server renders traces correctly, since that's Langfuse's code, not ours.

**Q: What real, unanticipated detail did running this for real surface?**
A: The request landed at `POST /api/public/otel/v1/traces`, not the
older `/api/public/ingestion` batch endpoint some Langfuse docs
reference. The installed SDK (Langfuse 4.x) is OpenTelemetry-native by
design — confirmed by inspecting the package source, which hardcodes the
OTel export path and ships `opentelemetry-exporter-otlp-proto-http` as a
dependency. The legacy ingestion endpoint is deprecated. Not a bug — a
version-specific transport detail worth knowing before assuming a
particular docs example's endpoint path is current.

**Q: Why does the demo call `get_client().flush()` before checking the
mock endpoint received anything?**
A: Langfuse batches spans (`BatchSpanProcessor`, ~512-span batches / 5s
delay by default) — without an explicit synchronous `flush()`, a
single-trace demo run could exit before the batch ever exports. `flush()`
blocks until delivery is confirmed, so there's no race between it
returning and checking `received_requests`.

**Q: Why does the demo only need to run through Curriculum Planner rather
than the full multi-topic loop?**
A: One real LLM call is enough to prove a callback fires. The graph
naturally stops at the existing `human_approval` interrupt right after,
which is a fast, convenient stopping point for this specific check.

**Q: Tracing is "on" but no traces show up anywhere — what are the most
likely causes?**
A: In rough order of likelihood: (1) `LANGFUSE_BASE_URL`/`LANGFUSE_HOST`
pointing at the wrong instance (e.g. a leftover localhost override when
using Langfuse Cloud), (2) `.env` never actually being loaded (check that
something in the app calls `load_dotenv()` — it's easy to set the right
values in `.env` and have them silently not apply), (3) rejected/invalid
API keys. None of these raise a Python exception — `langfuse_enabled()`
only checks that the keys are *present*, not that they're *valid* or
pointed at the *right host*.
