import os
from unittest.mock import patch


class TestTryA2AQuizDelegation:
    def test_a2a_disabled_returns_none(self):
        with patch.dict(os.environ, {"USE_A2A_QUIZ": "false"}):
            from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

            assert try_a2a_quiz_delegation("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_quiz_service_available", return_value=False)
    def test_returns_none_when_service_unavailable(self, mock_available):
        from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

        assert try_a2a_quiz_delegation("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_quiz_service_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.delegate_quiz_task")
    def test_returns_result_when_service_available(self, mock_delegate, mock_available):
        from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

        mock_delegate.return_value = {"status": "graded", "score": 0.85, "weak_areas": []}
        result = try_a2a_quiz_delegation("Topic", "Explanation", ["answer1"])
        assert result is not None
        assert result["status"] == "graded"

    @patch("learning_accelerator.a2a_services.a2a_client.is_quiz_service_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.delegate_quiz_task")
    def test_returns_none_on_delegation_error(self, mock_delegate, mock_available):
        from learning_accelerator.agents.progress_coach import try_a2a_quiz_delegation

        mock_delegate.return_value = {"error": "Service crashed"}
        assert try_a2a_quiz_delegation("Topic", "Explanation", []) is None


class TestTryStudyBuddyAssistance:
    def test_disabled_by_env_var_returns_none(self):
        with patch.dict(os.environ, {"USE_STUDY_BUDDY": "false"}):
            from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

            assert try_study_buddy_assistance("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_study_buddy_available", return_value=False)
    def test_returns_none_when_service_down(self, mock_avail):
        from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

        assert try_study_buddy_assistance("Topic", "Explanation", []) is None

    @patch("learning_accelerator.a2a_services.a2a_client.is_study_buddy_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.request_study_assistance")
    def test_returns_assistance_text_when_available(self, mock_assist, mock_avail):
        from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

        mock_assist.return_value = {
            "source": "crewai_study_buddy",
            "assistance": "Think of a checkpoint like a save file...",
            "status": "complete",
        }
        result = try_study_buddy_assistance("Topic", "Explanation", ["thread_id"])
        assert result is not None
        assert "save file" in result

    @patch("learning_accelerator.a2a_services.a2a_client.is_study_buddy_available", return_value=True)
    @patch("learning_accelerator.a2a_services.a2a_client.request_study_assistance")
    def test_returns_none_on_error_response(self, mock_assist, mock_avail):
        from learning_accelerator.agents.progress_coach import try_study_buddy_assistance

        mock_assist.return_value = {"status": "error", "error": "CrewAI crashed"}
        assert try_study_buddy_assistance("Topic", "Explanation", []) is None
