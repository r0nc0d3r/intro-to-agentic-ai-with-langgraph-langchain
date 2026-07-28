class TestQuizAgentCard:
    def test_agent_card_has_required_fields(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.name
        assert QUIZ_AGENT_CARD.url
        assert QUIZ_AGENT_CARD.version
        assert QUIZ_AGENT_CARD.skills
        assert len(QUIZ_AGENT_CARD.skills) > 0

    def test_agent_card_url_is_port_9001(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert "9001" in QUIZ_AGENT_CARD.url

    def test_skill_has_required_fields(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        skill = QUIZ_AGENT_CARD.skills[0]
        assert skill.id
        assert skill.name
        assert skill.description
        assert len(skill.description) > 20

    def test_skill_has_examples(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        skill = QUIZ_AGENT_CARD.skills[0]
        assert skill.examples
        assert len(skill.examples) >= 1

    def test_skill_id_is_correct(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.skills[0].id == "generate_and_grade_quiz"
