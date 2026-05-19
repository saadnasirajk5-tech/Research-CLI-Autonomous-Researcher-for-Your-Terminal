"""
Configuration for Research-CLI.

All settings are centralized here for easy tuning.
Small models need tighter constraints than GPT-4.

Prompts are loaded from template files in prompts/ directory.
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
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            from research_cli.prompts import load_prompt
            self.system_prompt = load_prompt("planner")


@dataclass
class ResearcherConfig:
    """Researcher agent settings."""
    max_results_per_search: int = 5
    max_pages_to_scrape: int = 3
    max_tokens_per_page: int = 4000
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            from research_cli.prompts import load_prompt
            self.system_prompt = load_prompt("researcher")


@dataclass
class CriticConfig:
    """Critic agent settings."""
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            from research_cli.prompts import load_prompt
            self.system_prompt = load_prompt("critic")


@dataclass
class WriterConfig:
    """Writer agent settings."""
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            from research_cli.prompts import load_prompt
            self.system_prompt = load_prompt("writer")


@dataclass
class ValidatorConfig:
    """Self-correction validator settings."""
    max_retries: int = 2
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            from research_cli.prompts import load_prompt
            self.system_prompt = load_prompt("validator")


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
