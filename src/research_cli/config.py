"""
Configuration for Research-CLI.

All settings are centralized here for easy tuning.
Small models need tighter constraints than GPT-4.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """LLM model settings."""
    provider: str = "ollama"
    model_name: str = os.getenv("RESEARCH_MODEL", "qwen2.5:3b")
    temperature: float = 0.1
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class PlannerConfig:
    """Planner agent settings."""
    max_subtasks: int = 8
    min_subtasks: int = 3
    system_prompt: str = """You are a Research Planner. Break the user's query into a directed acyclic graph (DAG) of independent sub-tasks.

Rules:
1. Each sub-task must be specific and answerable independently.
2. Sub-tasks should cover different angles: facts, comparisons, recent data, opposing views.
3. Output ONLY valid JSON with no markdown formatting.
4. Each sub-task needs: id, description, search_query, dependencies (list of task IDs this depends on).

Example output:
{{
  "subtasks": [
    {{"id": "t1", "description": "Find historical background on X", "search_query": "X history timeline key events", "dependencies": []}},
    {{"id": "t2", "description": "Compare viewpoints on Y", "search_query": "Y debate pros cons different perspectives", "dependencies": []}}
  ]
}}"""


@dataclass
class ResearcherConfig:
    """Researcher agent settings."""
    max_results_per_search: int = 5
    max_pages_to_scrape: int = 3
    max_tokens_per_page: int = 4000
    system_prompt: str = """You are a Research Specialist. Given a sub-task and search results, extract the most relevant facts, statistics, and quotes.

Rules:
1. Extract ONLY information directly supported by the provided sources.
2. Include citations (source URL) for every claim.
3. Be specific: prefer numbers, dates, names over vague statements.
4. If the sources don't contain useful info, say "INSUFFICIENT_DATA".
5. Output ONLY valid JSON with no markdown formatting.

Output format:
{{
  "findings": [
    {{"claim": "specific fact", "source_url": "https://...", "confidence": "high|medium|low"}}
  ],
  "summary": "brief summary of what was found",
  "needs_more_research": true/false
}}"""


@dataclass
class CriticConfig:
    """Critic agent settings."""
    system_prompt: str = """You are a Research Critic. Audit the researcher's findings against the original sub-task.

Check:
1. Does the finding directly address the sub-task?
2. Are there citations for each claim?
3. Is the data specific (numbers, dates, names) or vague?
4. Are there any unsupported claims?

Output ONLY valid JSON:
{{
  "passed": true/false,
  "score": 0-10,
  "issues": ["list of problems"],
  "feedback": "specific instructions for improvement if failed"
}}"""


@dataclass
class WriterConfig:
    """Writer agent settings."""
    system_prompt: str = """You are a Research Writer. Synthesize verified findings into a comprehensive report.

Rules:
1. Structure: Executive Summary, Key Findings, Detailed Analysis, Sources.
2. Every claim must be cited with its source.
3. Acknowledge uncertainties and conflicting data.
4. Write in clear, professional prose.
5. Do NOT invent information not present in the findings."""


@dataclass
class ValidatorConfig:
    """Self-correction validator settings."""
    max_retries: int = 2
    system_prompt: str = """You are a Fact Validator. Check if a claim is supported by the provided source text.

Output ONLY valid JSON:
{{
  "supported": true/false,
  "reason": "explanation",
  "corrected_claim": "rewritten claim if not supported, or empty if supported"
}}"""


@dataclass
class StorageConfig:
    """Persistence settings."""
    data_dir: Path = Path(os.getenv("RESEARCH_DATA_DIR", Path.home() / ".research-cli"))
    findings_dir: Path = field(default=None)
    sessions_dir: Path = field(default=None)
    chroma_dir: Path = field(default=None)

    def __post_init__(self):
        if self.findings_dir is None:
            self.findings_dir = self.data_dir / "findings"
        if self.sessions_dir is None:
            self.sessions_dir = self.data_dir / "sessions"
        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"


@dataclass
class UICConfig:
    """CLI UI settings."""
    show_cost: bool = True
    show_progress: bool = True
    theme: str = "dark"


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    researcher: ResearcherConfig = field(default_factory=ResearcherConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ui: UICConfig = field(default_factory=UICConfig)

    def initialize(self):
        """Create necessary directories."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.findings_dir.mkdir(parents=True, exist_ok=True)
        self.storage.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.storage.chroma_dir.mkdir(parents=True, exist_ok=True)
