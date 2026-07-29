from __future__ import annotations

import os

from langfuse.langchain import CallbackHandler


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_handler() -> CallbackHandler | None:
    if not langfuse_enabled():
        return None
    return CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"])


def get_langfuse_config(session_id: str) -> dict:
    config: dict = {"configurable": {"thread_id": session_id}}

    handler = get_langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
        config["metadata"] = {"langfuse_session_id": session_id}

    return config


def flush_langfuse() -> None:
    """Flush any pending Langfuse events before process exit.

    Langfuse batches/exports spans asynchronously; call this at the end
    of a script so all traces are actually sent before the process ends.
    No-op if Langfuse isn't configured.
    """
    if not langfuse_enabled():
        return

    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass  # best-effort flush, don't crash on exit
