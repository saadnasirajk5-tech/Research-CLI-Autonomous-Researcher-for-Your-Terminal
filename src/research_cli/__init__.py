"""
Research-CLI: Autonomous deep research orchestrator for local small models.

Zero API costs. Full transparency. Self-correcting research pipeline.

Usage:
    from research_cli import ResearchGraph, Config

    config = Config()
    graph = ResearchGraph(config)
    result = graph.run("What are the latest advances in fusion energy?")
    print(result["report"])
"""

from research_cli.config import Config
from research_cli.graph.workflow import ResearchGraph
from research_cli.graph.state import ResearchState, SubTask, Finding, TaskResult
from research_cli.main import main

__all__ = [
    "Config",
    "ResearchGraph",
    "ResearchState",
    "SubTask",
    "Finding",
    "TaskResult",
    "main",
]

__version__ = "0.1.0"
