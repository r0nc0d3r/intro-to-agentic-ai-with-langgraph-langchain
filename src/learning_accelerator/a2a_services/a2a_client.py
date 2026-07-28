from __future__ import annotations

import json
import os
import uuid

import httpx

QUIZ_SERVICE_URL = os.getenv("QUIZ_SERVICE_URL", "http://localhost:9001")
STUDY_BUDDY_URL = os.getenv("STUDY_BUDDY_URL", "http://localhost:9002")
DEFAULT_TIMEOUT = 120.0


def discover_agent(base_url: str) -> dict:
    card_url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
    try:
        response = httpx.get(card_url, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[A2A Client] Cannot reach {card_url}: {e}")
        return {}


def send_task(
    base_url: str,
    message_text: str,
    message_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": message_id or str(uuid.uuid4()),
                "kind": "message",
                "parts": [{"kind": "text", "text": message_text}],
            },
        },
    }

    url = base_url.rstrip("/") + "/"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return {"error": f"A2A task failed: {data['error']}"}

        result = data.get("result", {})

        # Agent responded directly with a Message (result.kind == "message").
        for part in result.get("parts", []):
            if part.get("kind") == "text":
                try:
                    return json.loads(part["text"])
                except json.JSONDecodeError:
                    return {"text": part["text"]}

        # Agent responded with a Task (result.status.message.parts / artifacts).
        artifacts = result.get("artifacts", [])
        if artifacts:
            for part in artifacts[0].get("parts", []):
                if part.get("kind") == "text":
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        return {"text": part["text"]}

        status = result.get("status", {})
        if status:
            msg = status.get("message", {})
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    try:
                        return json.loads(part["text"])
                    except json.JSONDecodeError:
                        return {"text": part["text"]}

        return result

    except httpx.TimeoutException:
        return {"error": f"A2A service timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to A2A service at {url}"}
    except Exception as e:
        return {"error": f"A2A task failed: {type(e).__name__}: {e}"}


def delegate_quiz_task(
    topic: str,
    explanation: str,
    answers: list[str] | None = None,
    quiz_service_url: str = QUIZ_SERVICE_URL,
) -> dict:
    payload = json.dumps(
        {"topic": topic, "explanation": explanation, "answers": answers or []}
    )
    return send_task(quiz_service_url, payload)


def is_quiz_service_available(quiz_service_url: str = QUIZ_SERVICE_URL) -> bool:
    return bool(discover_agent(quiz_service_url))


def request_study_assistance(
    topic: str,
    explanation: str,
    weak_areas: list[str] | None = None,
    study_buddy_url: str = STUDY_BUDDY_URL,
) -> dict:
    payload = json.dumps(
        {"topic": topic, "explanation": explanation, "weak_areas": weak_areas or []}
    )
    return send_task(study_buddy_url, payload, timeout=180.0)


def is_study_buddy_available(study_buddy_url: str = STUDY_BUDDY_URL) -> bool:
    return bool(discover_agent(study_buddy_url))
