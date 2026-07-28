import json

import httpx
from unittest.mock import MagicMock, patch


class TestDiscoverAgent:
    @patch("learning_accelerator.a2a_services.a2a_client.httpx.get")
    def test_returns_card_on_success(self, mock_get):
        from learning_accelerator.a2a_services.a2a_client import discover_agent

        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "Test Agent", "url": "http://localhost:9001/"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = discover_agent("http://localhost:9001")
        assert result["name"] == "Test Agent"

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_connection_error(self, mock_get):
        from learning_accelerator.a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.ConnectError("Connection refused")
        assert discover_agent("http://localhost:9001") == {}

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.get")
    def test_returns_empty_dict_on_timeout(self, mock_get):
        from learning_accelerator.a2a_services.a2a_client import discover_agent

        mock_get.side_effect = httpx.TimeoutException("Timed out")
        assert discover_agent("http://localhost:9001") == {}


class TestSendTask:
    @patch("learning_accelerator.a2a_services.a2a_client.httpx.post")
    def test_returns_parsed_result_on_success(self, mock_post):
        from learning_accelerator.a2a_services.a2a_client import send_task

        quiz_result = {"status": "questions_ready", "topic": "Test Topic", "questions": []}
        mock_response = MagicMock()
        # The a2a-sdk's message/send returns the agent's reply as a top-level
        # Message object (kind="message"), not wrapped in Task artifacts.
        mock_response.json.return_value = {
            "result": {
                "kind": "message",
                "role": "agent",
                "parts": [{"kind": "text", "text": json.dumps(quiz_result)}],
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = send_task("http://localhost:9001", json.dumps({"topic": "Test"}))
        assert result["status"] == "questions_ready"

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_connection_refused(self, mock_post):
        from learning_accelerator.a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.ConnectError("Connection refused")
        result = send_task("http://localhost:9001", "{}")
        assert "error" in result
        assert "connect" in result["error"].lower()

    @patch("learning_accelerator.a2a_services.a2a_client.httpx.post")
    def test_returns_error_on_timeout(self, mock_post):
        from learning_accelerator.a2a_services.a2a_client import send_task

        mock_post.side_effect = httpx.TimeoutException("Timed out")
        result = send_task("http://localhost:9001", "{}", timeout=1.0)
        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestDelegateQuizTask:
    @patch("learning_accelerator.a2a_services.a2a_client.send_task")
    def test_sends_correct_payload(self, mock_send):
        from learning_accelerator.a2a_services.a2a_client import delegate_quiz_task

        mock_send.return_value = {"status": "graded", "score": 0.8}
        delegate_quiz_task(topic="LangGraph", explanation="Nodes and edges...", answers=["my answer"])

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["topic"] == "LangGraph"
        assert "explanation" in payload
        assert "answers" in payload

    @patch("learning_accelerator.a2a_services.a2a_client.send_task")
    def test_empty_answers_sends_empty_list(self, mock_send):
        from learning_accelerator.a2a_services.a2a_client import delegate_quiz_task

        mock_send.return_value = {"status": "questions_ready"}
        delegate_quiz_task("Topic", "Explanation", answers=None)

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["answers"] == []


class TestIsQuizServiceAvailable:
    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_returns_true_when_card_available(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_quiz_service_available

        mock_discover.return_value = {"name": "Quiz Service"}
        assert is_quiz_service_available() is True

    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_returns_false_when_service_down(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_quiz_service_available

        mock_discover.return_value = {}
        assert is_quiz_service_available() is False


class TestStudyBuddyClient:
    @patch("learning_accelerator.a2a_services.a2a_client.send_task")
    def test_request_study_assistance_sends_correct_payload(self, mock_send):
        from learning_accelerator.a2a_services.a2a_client import request_study_assistance

        mock_send.return_value = {"source": "crewai_study_buddy", "assistance": "...", "status": "complete"}
        request_study_assistance(topic="LangGraph", explanation="...", weak_areas=["checkpointing"])

        payload = json.loads(mock_send.call_args[0][1])
        assert payload["topic"] == "LangGraph"
        assert payload["weak_areas"] == ["checkpointing"]
        assert "explanation" in payload

    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_is_study_buddy_available_true(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_study_buddy_available

        mock_discover.return_value = {"name": "CrewAI Study Buddy"}
        assert is_study_buddy_available() is True

    @patch("learning_accelerator.a2a_services.a2a_client.discover_agent")
    def test_is_study_buddy_available_false(self, mock_discover):
        from learning_accelerator.a2a_services.a2a_client import is_study_buddy_available

        mock_discover.return_value = {}
        assert is_study_buddy_available() is False
