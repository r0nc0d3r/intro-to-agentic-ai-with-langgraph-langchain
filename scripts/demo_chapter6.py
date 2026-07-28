"""Manual run: verify the Langfuse callback wiring fires real network
requests when configured, and gracefully omits callbacks when it isn't.

There's no live Langfuse instance in this environment — self-hosting the
full v3 stack (postgres+clickhouse+redis+minio+worker+web) is too heavy
to stand up just for this check (see docker-compose.langfuse.yml if you
want to run one yourself). Instead, this script runs a tiny local HTTP
server that stands in for Langfuse's ingestion endpoint and records
whether any POST requests arrive — proving our CallbackHandler wiring
genuinely fires real requests at LANGFUSE_HOST, not silently no-oping.

One real LLM call (Curriculum Planner) is enough to prove a traced call
fires; the graph then hits the existing human_approval interrupt, which
is a natural, fast stopping point for this specific check.

Requires either a running Ollama instance (default, see .env.example) or
ANTHROPIC_API_KEY / OPENAI_API_KEY with LLM_PROVIDER set accordingly.
"""

from __future__ import annotations

import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from langfuse import get_client

from learning_accelerator.graph.state import initial_state
from learning_accelerator.graph.workflow import build_graph
from learning_accelerator.observability.langfuse_setup import get_langfuse_config

received_requests: list[tuple[str, int]] = []


class _MockIngestionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        received_requests.append((self.path, len(body)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"successes": [], "errors": []}')

    def log_message(self, format: str, *args: object) -> None:
        pass  # silence default per-request stderr logging


def main() -> None:
    server = HTTPServer(("localhost", 0), _MockIngestionHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test-chapter6"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test-chapter6"
    os.environ["LANGFUSE_HOST"] = f"http://localhost:{port}"

    session_id = str(uuid.uuid4())
    config = get_langfuse_config(session_id)
    assert "callbacks" in config, "expected Langfuse callback to be wired in when keys are set"

    graph = build_graph()
    state = initial_state(goal="Learn the basics of LangGraph", session_id=session_id)
    result = graph.invoke(state, config=config)
    assert "__interrupt__" in result, "expected the graph to stop at the human_approval interrupt"

    get_client().flush()
    server.shutdown()

    print(f"Mock Langfuse endpoint received {len(received_requests)} request(s).")
    for path, size in received_requests:
        print(f"  POST {path} ({size} bytes)")
    assert len(received_requests) > 0, "expected at least one request to reach the mock endpoint"

    print("\n--- now verifying graceful no-op when Langfuse is NOT configured ---")
    del os.environ["LANGFUSE_PUBLIC_KEY"]
    del os.environ["LANGFUSE_SECRET_KEY"]
    plain_config = get_langfuse_config(str(uuid.uuid4()))
    assert "callbacks" not in plain_config
    print("Confirmed: no 'callbacks' key when Langfuse credentials aren't set.")


if __name__ == "__main__":
    main()
