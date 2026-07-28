import json


class TestStudyBuddyAgentCard:
    def test_agent_card_has_required_fields(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.name
        assert STUDY_BUDDY_CARD.url
        assert STUDY_BUDDY_CARD.version
        assert STUDY_BUDDY_CARD.skills
        assert len(STUDY_BUDDY_CARD.skills) > 0

    def test_agent_card_url_is_port_9002(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert "9002" in STUDY_BUDDY_CARD.url

    def test_skill_id_is_correct(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.skills[0].id == "supplementary_study_assistance"

    def test_skill_mentions_crewai(self):
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        card_text = (STUDY_BUDDY_CARD.description + STUDY_BUDDY_CARD.skills[0].description).lower()
        assert "crewai" in card_text

    def test_different_port_from_quiz_service(self):
        from learning_accelerator.a2a_services.quiz_service import QUIZ_AGENT_CARD
        from learning_accelerator.crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.url != QUIZ_AGENT_CARD.url


class TestBuildStudyBuddyCrew:
    def test_returns_crew_object(self):
        from crewai import Crew

        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="LangGraph Checkpointing",
            explanation="Checkpoints persist state after each step...",
            weak_areas=["thread_id"],
        )
        assert isinstance(crew, Crew)

    def test_crew_has_one_agent(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        assert len(crew.agents) == 1

    def test_crew_has_one_task(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        assert len(crew.tasks) == 1

    def test_agent_has_study_buddy_role(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        agent = crew.agents[0]
        assert "study" in agent.role.lower() or "buddy" in agent.role.lower()

    def test_task_description_contains_topic(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="LangGraph Reducers", explanation="Reducers control merging...", weak_areas=[]
        )
        assert "LangGraph Reducers" in crew.tasks[0].description

    def test_task_description_contains_weak_areas(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew(
            topic="Checkpointing", explanation="...", weak_areas=["thread_id", "SqliteSaver"]
        )
        desc = crew.tasks[0].description
        assert "thread_id" in desc or "SqliteSaver" in desc

    def test_agent_has_topic_analyser_tool(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew = build_study_buddy_crew("Topic", "Explanation", [])
        tool_names = [type(t).__name__ for t in crew.agents[0].tools]
        assert "TopicAnalyserTool" in tool_names

    def test_different_topics_create_different_tasks(self):
        from learning_accelerator.crewai_agent.study_buddy import build_study_buddy_crew

        crew1 = build_study_buddy_crew("Checkpointing", "Explanation 1", [])
        crew2 = build_study_buddy_crew("Reducers", "Explanation 2", [])
        assert crew1.tasks[0].description != crew2.tasks[0].description


class TestTopicAnalyserTool:
    def test_returns_json_string(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = tool._run(topic="LangGraph Checkpointing", weak_areas=["thread_id"])
        assert isinstance(json.loads(result), dict)

    def test_result_contains_topic(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Reducers", weak_areas=[]))
        assert result["topic"] == "Reducers"

    def test_result_has_required_keys(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="Checkpointing", weak_areas=["thread_id"]))
        for key in ["topic", "focus_areas", "suggested_approach", "study_tip"]:
            assert key in result, f"Missing key: {key}"

    def test_weak_areas_appear_in_focus_areas(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(
            tool._run(topic="Checkpointing", weak_areas=["thread_id", "SqliteSaver"])
        )
        assert "thread_id" in result["focus_areas"]
        assert "SqliteSaver" in result["focus_areas"]

    def test_empty_weak_areas_uses_fallback(self):
        from learning_accelerator.crewai_agent.study_buddy import TopicAnalyserTool

        tool = TopicAnalyserTool()
        result = json.loads(tool._run(topic="LangGraph Basics", weak_areas=[]))
        assert len(result["focus_areas"]) > 0
