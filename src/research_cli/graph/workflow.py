"""
LangGraph Workflow - The Research Orchestration Graph.

Defines the state machine that orchestrates all agents:
Planner -> [Researchers in parallel] -> Critic -> [Validator loop] -> Writer

Uses LangGraph's StateGraph for explicit state transitions.
Each node is an agent, edges define the flow.
"""

import uuid
from typing import Any
from langgraph.graph import StateGraph, END
from research_cli.config import Config
from research_cli.graph.state import ResearchState
from research_cli.agents.planner import PlannerAgent
from research_cli.agents.researcher import ResearcherAgent
from research_cli.agents.critic import CriticAgent
from research_cli.agents.validator import ValidatorAgent
from research_cli.agents.writer import WriterAgent
from research_cli.storage.persistence import StatePersistence, save_finding_to_disk
from research_cli.tools.rag import LocalRAG


class ResearchGraph:
    """
    The main orchestration graph for deep research.

    Workflow:
    1. PLANNER: Break query into sub-tasks (DAG)
    2. RESEARCHERS: Execute sub-tasks in parallel
    3. CRITIC: Audit each result
    4. VALIDATOR: Fact-check claims against sources
    5. WRITER: Synthesize final report

    State is persisted after each step for crash recovery.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the graph with all agents."""
        self.config = config or Config()
        self.config.initialize()
        self.planner = PlannerAgent(self.config)
        self.researcher = ResearcherAgent(self.config)
        self.critic = CriticAgent(self.config)
        self.validator = ValidatorAgent(self.config)
        self.writer = WriterAgent(self.config)
        self.persistence = StatePersistence(self.config)
        self.rag = LocalRAG(self.config)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(ResearchState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("writer", self._writer_node)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "research")
        workflow.add_edge("research", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._should_revise,
            {"revise": "research", "write": "validate"},
        )
        workflow.add_edge("validate", "writer")
        workflow.add_edge("writer", END)
        return workflow.compile()

    def run(self, query: str, session_id: str | None = None) -> dict:
        """
        Execute the full research workflow.

        Parameters:
            query: The user's research question.
            session_id: Optional session ID for resuming.

        Returns:
            The final state dict with the report.
        """
        if not session_id:
            session_id = str(uuid.uuid4())[:12]
        initial_state = {
            "query": query,
            "subtasks": [],
            "task_results": [],
            "findings": [],
            "report": "",
            "current_task": "",
            "all_tasks_complete": False,
            "needs_revision": False,
            "revision_task_id": None,
            "error": None,
            "session_id": session_id,
            "tokens_used": 0,
            "cost_estimate_usd": 0.0,
        }
        result = self.graph.invoke(initial_state)
        self.persistence.save_state(result)
        return result

    def _planner_node(self, state: ResearchState) -> dict:
        """Planner node: break query into sub-tasks."""
        subtasks = self.planner.plan(state["query"])
        state["subtasks"] = subtasks
        self.persistence.save_state(state)
        return {"subtasks": subtasks}

    def _research_node(self, state: ResearchState) -> dict:
        """Research node: execute sub-tasks (parallel in concept, sequential for small models)."""
        task_results = []
        for task in state["subtasks"]:
            state["current_task"] = task["id"]
            result = self.researcher.research(
                task_id=task["id"],
                description=task["description"],
                search_query=task["search_query"],
            )
            result["description"] = task["description"]
            result["retry_count"] = 0
            task_results.append(result)
            for finding in result.get("findings", []):
                save_finding_to_disk(finding, self.config)
                self.rag.add_finding(
                    finding.get("claim", ""),
                    task_id=task["id"],
                    source_url=finding.get("source_url", ""),
                )
        state["task_results"] = task_results
        self.persistence.save_state(state)
        return {"task_results": task_results}

    def _critic_node(self, state: ResearchState) -> dict:
        """Critic node: audit all task results."""
        needs_revision = False
        revision_task_id = None
        updated_results = []
        for result in state["task_results"]:
            task_desc = result.get("description", "")
            evaluation = self.critic.evaluate(task_desc, result)
            result["critic_passed"] = evaluation["passed"]
            result["critic_score"] = evaluation["score"]
            result["critic_feedback"] = evaluation["feedback"]
            if not evaluation["passed"] and result.get("retry_count", 0) < self.config.validator.max_retries:
                needs_revision = True
                revision_task_id = result["task_id"]
                result["retry_count"] = result.get("retry_count", 0) + 1
            updated_results.append(result)
        state["task_results"] = updated_results
        state["needs_revision"] = needs_revision
        state["revision_task_id"] = revision_task_id
        self.persistence.save_state(state)
        return {
            "task_results": updated_results,
            "needs_revision": needs_revision,
            "revision_task_id": revision_task_id,
        }

    def _should_revise(self, state: ResearchState) -> str:
        """Conditional edge: decide whether to revise or move to validation."""
        if state.get("needs_revision", False):
            return "revise"
        return "write"

    def _validate_node(self, state: ResearchState) -> dict:
        """Validator node: fact-check all findings."""
        all_findings = []
        for result in state["task_results"]:
            findings = result.get("findings", [])
            validated = self.validator.validate_batch(findings)
            result["findings"] = validated
            all_findings.extend(validated)
        state["task_results"] = state["task_results"]
        state["findings"] = all_findings
        state["all_tasks_complete"] = True
        self.persistence.save_state(state)
        return {"findings": all_findings, "all_tasks_complete": True}

    def _writer_node(self, state: ResearchState) -> dict:
        """Writer node: generate the final report."""
        report = self.writer.write_report(state["query"], state["task_results"])
        state["report"] = report
        self.persistence.save_state(state)
        return {"report": report}
