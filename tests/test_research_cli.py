"""
Tests for Research-CLI.

Tests cover:
1. Config initialization and prompt loading
2. Planner parsing
3. Researcher output parsing
4. Critic evaluation
5. Validator logic
6. State persistence
7. RAG storage
8. Scraper
9. Search tool
10. Graph workflow (mocked)
11. Parallel execution
12. Writer with validated findings
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from research_cli.config import Config
from research_cli.graph.state import ResearchState
from research_cli.storage.persistence import StatePersistence, save_finding_to_disk
from research_cli.tools.search import search_web


class TestConfig:
    """Test configuration initialization."""

    def test_default_config(self):
        config = Config()
        assert config.model.model_name == "qwen2.5:3b"
        assert config.planner.max_subtasks == 8
        assert config.validator.max_retries == 2

    def test_initialize_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir) / "test_data"
            config.initialize()
            assert config.storage.data_dir.exists()
            assert config.storage.findings_dir.exists()
            assert config.storage.sessions_dir.exists()
            assert config.storage.chroma_dir.exists()

    def test_prompts_loaded_from_files(self):
        config = Config()
        assert len(config.planner.system_prompt) > 50
        assert len(config.researcher.system_prompt) > 50
        assert len(config.critic.system_prompt) > 50
        assert len(config.writer.system_prompt) > 50
        assert len(config.validator.system_prompt) > 50


class TestPromptLoader:
    """Test the prompt loading system."""

    def test_load_planner_prompt(self):
        from research_cli.prompts import load_prompt
        prompt = load_prompt("planner")
        assert "Research Planner" in prompt
        assert "subtasks" in prompt

    def test_load_researcher_prompt(self):
        from research_cli.prompts import load_prompt
        prompt = load_prompt("researcher")
        assert "Research Specialist" in prompt
        assert "findings" in prompt

    def test_load_critic_prompt(self):
        from research_cli.prompts import load_prompt
        prompt = load_prompt("critic")
        assert "Research Critic" in prompt
        assert "passed" in prompt

    def test_load_writer_prompt(self):
        from research_cli.prompts import load_prompt
        prompt = load_prompt("writer")
        assert "Research Writer" in prompt
        assert "Executive Summary" in prompt

    def test_load_validator_prompt(self):
        from research_cli.prompts import load_prompt
        prompt = load_prompt("validator")
        assert "Fact Validator" in prompt
        assert "supported" in prompt

    def test_load_nonexistent_prompt(self):
        from research_cli.prompts import load_prompt
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt")


class TestPlannerParsing:
    """Test the planner's JSON parsing logic."""

    def test_valid_json_parsing(self):
        from research_cli.agents.planner import PlannerAgent
        agent = PlannerAgent()
        raw = json.dumps({
            "subtasks": [
                {"id": "t1", "description": "Test task", "search_query": "test", "dependencies": []}
            ]
        })
        result = agent._parse_response(raw)
        assert len(result) == 1
        assert result[0]["id"] == "t1"

    def test_markdown_wrapped_json(self):
        from research_cli.agents.planner import PlannerAgent
        agent = PlannerAgent()
        raw = '```json\n{"subtasks": [{"id": "t1", "description": "x", "search_query": "y", "dependencies": []}]}\n```'
        result = agent._parse_response(raw)
        assert len(result) == 1

    def test_fallback_plan(self):
        from research_cli.agents.planner import PlannerAgent
        agent = PlannerAgent()
        result = agent._fallback_plan("some invalid response")
        assert len(result) == 3
        assert result[0]["id"] == "t1"


class TestResearcherParsing:
    """Test the researcher's output parsing."""

    def test_valid_findings(self):
        from research_cli.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        raw = json.dumps({
            "findings": [
                {"claim": "Test claim", "source_url": "https://example.com", "confidence": "high"}
            ],
            "summary": "Test summary",
            "needs_more_research": False,
        })
        sources = [{"url": "https://example.com", "title": "Example"}]
        result = agent._parse_response(raw, "t1", sources)
        assert len(result["findings"]) == 1
        assert result["findings"][0]["claim"] == "Test claim"
        assert result["findings"][0]["task_id"] == "t1"

    def test_fallback_result(self):
        from research_cli.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        sources = [{"url": "https://example.com", "title": "Example"}]
        result = agent._fallback_result("raw text response", "t1", sources)
        assert result["task_id"] == "t1"
        assert len(result["findings"]) == 1


class TestCriticParsing:
    """Test the critic's evaluation parsing."""

    def test_valid_evaluation(self):
        from research_cli.agents.critic import CriticAgent
        agent = CriticAgent()
        raw = json.dumps({
            "passed": True,
            "score": 8,
            "issues": [],
            "feedback": "",
        })
        result = agent._parse_response(raw)
        assert result["passed"] is True
        assert result["score"] == 8

    def test_score_clamping(self):
        from research_cli.agents.critic import CriticAgent
        agent = CriticAgent()
        raw = json.dumps({"passed": False, "score": 15, "issues": ["x"], "feedback": "y"})
        result = agent._parse_response(raw)
        assert result["score"] == 10

    def test_fallback_evaluation(self):
        from research_cli.agents.critic import CriticAgent
        agent = CriticAgent()
        result = agent._fallback_evaluate("short")
        assert result["passed"] is False
        assert result["score"] == 2


class TestValidatorParsing:
    """Test the validator's response parsing."""

    def test_supported_claim(self):
        from research_cli.agents.validator import ValidatorAgent
        agent = ValidatorAgent()
        raw = json.dumps({
            "supported": True,
            "reason": "Source confirms the claim.",
            "corrected_claim": "",
        })
        result = agent._parse_response(raw)
        assert result["supported"] is True

    def test_unsupported_claim(self):
        from research_cli.agents.validator import ValidatorAgent
        agent = ValidatorAgent()
        raw = json.dumps({
            "supported": False,
            "reason": "Source says the opposite.",
            "corrected_claim": "Corrected version.",
        })
        result = agent._parse_response(raw)
        assert result["supported"] is False
        assert result["corrected_claim"] == "Corrected version."

    def test_empty_url(self):
        from research_cli.agents.validator import ValidatorAgent
        agent = ValidatorAgent()
        result = agent.validate_claim("some claim", "")
        assert result["supported"] is False


class TestStatePersistence:
    """Test JSON-based state persistence."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            persistence = StatePersistence(config)
            state = {
                "query": "test query",
                "session_id": "test-123",
                "subtasks": [{"id": "t1", "description": "test", "search_query": "test", "dependencies": []}],
                "task_results": [],
                "findings": [],
                "report": "",
                "current_task": "",
                "all_tasks_complete": False,
                "needs_revision": False,
                "revision_task_id": None,
                "error": None,
                "tokens_used": 0,
                "cost_estimate_usd": 0.0,
            }
            session_id = persistence.save_state(state)
            assert session_id == "test-123"
            loaded = persistence.load_state("test-123")
            assert loaded is not None
            assert loaded["query"] == "test query"

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            persistence = StatePersistence(config)
            assert persistence.load_state("nonexistent") is None

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            persistence = StatePersistence(config)
            state = {"query": "test", "session_id": "s1", "subtasks": [], "task_results": [], "findings": [], "report": "", "current_task": "", "all_tasks_complete": False, "needs_revision": False, "revision_task_id": None, "error": None, "tokens_used": 0, "cost_estimate_usd": 0.0}
            persistence.save_state(state)
            sessions = persistence.list_sessions()
            assert len(sessions) >= 1

    def test_delete_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            persistence = StatePersistence(config)
            state = {"query": "test", "session_id": "del-1", "subtasks": [], "task_results": [], "findings": [], "report": "", "current_task": "", "all_tasks_complete": False, "needs_revision": False, "revision_task_id": None, "error": None, "tokens_used": 0, "cost_estimate_usd": 0.0}
            persistence.save_state(state)
            assert persistence.delete_session("del-1") is True
            assert persistence.load_state("del-1") is None

    def test_save_finding_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            finding = {"task_id": "t1", "claim": "test claim", "source_url": "https://x.com", "confidence": "high", "validated": False}
            path = save_finding_to_disk(finding, config)
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["claim"] == "test claim"


class TestSearchTool:
    """Test the search tool."""

    @patch("research_cli.tools.search.DDGS_CLASS")
    def test_search_web(self, mock_ddgs):
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "Test Result", "body": "Test snippet", "href": "https://test.com"}
        ]
        mock_ddgs.return_value = mock_instance
        results = search_web("test query", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Test Result"

    def test_search_web_error(self):
        results = search_web("", max_results=1)
        assert len(results) == 1
        assert "Error" in results[0]["title"]


class TestScraper:
    """Test the scraper tool."""

    def test_empty_url(self):
        from research_cli.tools.scraper import scrape_page
        result = scrape_page("")
        assert "Empty URL" in result

    def test_fallback_scrape(self):
        from research_cli.tools.scraper import _html_to_text
        html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        text = _html_to_text(html)
        assert "Hello" in text
        assert "World" in text


class TestRAG:
    """Test the local RAG index."""

    def test_add_and_query_fallback(self):
        """Test keyword-based fallback when ChromaDB is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            from research_cli.tools.rag import LocalRAG
            rag = LocalRAG(config)
            rag.add_finding("The capital of France is Paris.", task_id="t1", source_url="https://example.com")
            results = rag.query("capital France Paris", n_results=1)
            assert len(results) >= 1
            assert "Paris" in results[0]["text"]

    def test_query_with_task_filter(self):
        """Test task filtering in fallback mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            from research_cli.tools.rag import LocalRAG
            rag = LocalRAG(config)
            rag.add_finding("Finding for task one", task_id="t1")
            rag.add_finding("Finding for task two", task_id="t2")
            results = rag.query("finding", n_results=5, task_id="t1")
            for r in results:
                assert r["task_id"] == "t1"


class TestGraphWorkflow:
    """Test the LangGraph workflow structure."""

    def test_graph_builds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = Path(tmpdir)
            config.initialize()
            from research_cli.graph.workflow import ResearchGraph
            graph = ResearchGraph(config)
            assert graph.graph is not None

    def test_state_no_operator_add(self):
        """Verify state uses replace semantics, not operator.add."""
        from research_cli.graph.state import ResearchState
        import typing
        hints = typing.get_type_hints(ResearchState)
        task_results_type = str(hints.get("task_results", ""))
        assert "operator" not in task_results_type.lower()
        assert "add" not in task_results_type.lower()

    def test_state_has_revision_fields(self):
        """Verify state has revision_task_id for targeted retries."""
        import typing
        from research_cli.graph.state import ResearchState
        hints = typing.get_type_hints(ResearchState)
        assert "revision_task_id" in hints
        assert "needs_revision" in hints

    def test_writer_uses_findings_not_task_results(self):
        """Verify writer node passes findings, not raw task_results."""
        import inspect
        from research_cli.graph.workflow import ResearchGraph
        source = inspect.getsource(ResearchGraph._writer_node)
        assert 'state["findings"]' in source
        assert 'state["task_results"]' not in source

    def test_research_uses_thread_pool(self):
        """Verify research node uses ThreadPoolExecutor for parallelism."""
        import inspect
        from research_cli.graph.workflow import ResearchGraph
        source = inspect.getsource(ResearchGraph._research_node)
        assert "ThreadPoolExecutor" in source

    def test_research_filters_by_revision_id(self):
        """Verify research node only retries the failed task."""
        import inspect
        from research_cli.graph.workflow import ResearchGraph
        source = inspect.getsource(ResearchGraph._research_node)
        assert "revision_task_id" in source or "revision_id" in source


class TestPackageExports:
    """Test that __init__.py exports the public API."""

    def test_exports_config(self):
        from research_cli import Config
        assert Config is not None

    def test_exports_research_graph(self):
        from research_cli import ResearchGraph
        assert ResearchGraph is not None

    def test_exports_state_types(self):
        from research_cli import ResearchState, SubTask, Finding, TaskResult
        assert ResearchState is not None
        assert SubTask is not None
        assert Finding is not None
        assert TaskResult is not None

    def test_exports_main(self):
        from research_cli import main
        assert callable(main)
