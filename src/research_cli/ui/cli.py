"""
Rich CLI UI - Beautiful terminal interface.

Uses Rich for colored output, progress bars, and panels.
Shows real-time progress, cost estimates, and verification status.
Makes the research process transparent and visually appealing.
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from research_cli.config import Config


console = Console()


class ResearchUI:
    """
    CLI interface for the research orchestrator.

    Provides:
    - Animated progress indicators
    - Cost comparison (vs GPT-4)
    - Real-time status updates
    - Formatted report output
    """

    GPT4_COST_PER_1K_TOKENS = 0.03
    GPT4_OUTPUT_COST_PER_1K_TOKENS = 0.06

    def __init__(self, config: Config | None = None):
        """Initialize the UI."""
        self.config = config or Config()
        self.config.initialize()

    def print_banner(self):
        """Print the startup banner."""
        banner = """
  ___  ___  _ __  _ __ ___   ___ _ __
 / _ \\/ _ \\| '_ \\| '__/ _ \\ / _ \\ '__|
|  __/ (_) | |_) | | | (_) |  __/ |
 \\___|\\___/| .__/|_|  \\___/ \\___|_|
           |_|
  Autonomous Deep Research for Local Models
  Zero API Costs. Full Transparency."""
        console.print(Panel(banner, style="bold cyan", border_style="cyan"))
        console.print()

    def print_cost_comparison(self, tokens_used: int):
        """
        Show how much the user saved vs using GPT-4.

        This is a key feature for the README screenshot.
        """
        if not self.config.ui.show_cost:
            return
        gpt4_input_cost = (tokens_used * 0.5) / 1000 * self.GPT4_COST_PER_1K_TOKENS
        gpt4_output_cost = (tokens_used * 0.5) / 1000 * self.GPT4_OUTPUT_COST_PER_1K_TOKENS
        gpt4_total = gpt4_input_cost + gpt4_output_cost
        table = Table(title="Cost Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan")
        table.add_column("Estimated Cost", style="green")
        table.add_row("GPT-4 (cloud)", f"${gpt4_total:.4f}")
        table.add_row("Local (Ollama)", "$0.0000")
        table.add_row("You saved", f"${gpt4_total:.4f}", style="bold green")
        console.print(table)
        console.print()

    def show_progress(self, stage: str, detail: str = ""):
        """Show a single progress step with spinner."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            TextColumn("[dim]{task.fields[detail]}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(stage, detail=detail)
            progress.advance(task)

    def show_research_progress(self, stages: list[tuple[str, str]]):
        """
        Show progress through all research stages.

        Parameters:
            stages: List of (stage_name, detail) tuples.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TextColumn("[dim]{task.fields[detail]}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            for stage_name, detail in stages:
                task = progress.add_task(stage_name, detail=detail, total=1.0)
                progress.update(task, completed=1.0)

    def print_task_status(self, task_id: str, description: str, status: str):
        """Print the status of a single research task."""
        status_colors = {
            "searching": "yellow",
            "scraping": "magenta",
            "analyzing": "blue",
            "validated": "green",
            "failed": "red",
            "revising": "orange3",
        }
        color = status_colors.get(status, "white")
        console.print(f"  [{color}]●[/{color}] [{color}]{status}[/{color}] {task_id}: {description}")

    def print_verification_result(self, claim: str, supported: bool, reason: str):
        """Print a single claim verification result."""
        if supported:
            console.print(f"    [green]✓ VERIFIED[/green] {claim[:80]}...")
        else:
            console.print(f"    [red]✗ FAILED[/red] {claim[:80]}...")
            console.print(f"      [dim]Reason: {reason}[/dim]")

    def print_report(self, report: str):
        """Print the final research report with markdown formatting."""
        console.print()
        console.print(Panel("Research Report", style="bold green", border_style="green"))
        console.print()
        console.print(Markdown(report))
        console.print()

    def print_finding_summary(self, findings: list[dict]):
        """Print a summary table of all findings."""
        table = Table(title="Findings Summary", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim")
        table.add_column("Claim", style="white")
        table.add_column("Status", style="green")
        table.add_column("Source", style="dim")
        for i, f in enumerate(findings, 1):
            status = "[green]Verified[/green]" if f.get("validated") else "[yellow]Unverified[/yellow]"
            table.add_row(
                str(i),
                f.get("claim", "")[:60],
                status,
                f.get("source_url", "")[:40],
            )
        console.print(table)
        console.print()

    def print_error(self, message: str):
        """Print an error message."""
        console.print(f"[bold red]Error:[/bold red] {message}")

    def print_session_info(self, session_id: str, tokens_used: int):
        """Print session information."""
        console.print(f"[dim]Session ID: {session_id}[/dim]")
        console.print(f"[dim]Tokens used: ~{tokens_used}[/dim]")
        console.print()

    def prompt_query(self) -> str:
        """Prompt the user for a research query."""
        console.print()
        query = console.input("[bold cyan]What would you like to research? [/bold cyan]")
        return query.strip()
