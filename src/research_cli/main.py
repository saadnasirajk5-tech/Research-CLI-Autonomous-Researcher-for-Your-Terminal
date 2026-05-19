"""
Research-CLI Main Entry Point.

The CLI orchestrator that ties everything together:
1. Parse arguments
2. Initialize config and UI
3. Run the research graph
4. Display results with cost comparison
"""

import argparse
import sys
from research_cli.config import Config
from research_cli.graph.workflow import ResearchGraph
from research_cli.ui.cli import ResearchUI
from research_cli.storage.persistence import StatePersistence


def main():
    """Main entry point for the research CLI."""
    parser = argparse.ArgumentParser(
        description="Research-CLI: Autonomous deep research on your laptop. Zero API costs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  research "What are the latest advances in fusion energy?"
  research "Compare React vs Vue vs Svelte in 2025" --model llama3.1:8b
  research --resume <session-id>
  research --list-sessions
        """,
    )
    parser.add_argument("query", nargs="?", help="The research question to investigate")
    parser.add_argument("--model", default=None, help="Ollama model to use (default: qwen2.5:3b)")
    parser.add_argument("--resume", metavar="SESSION_ID", help="Resume a saved session")
    parser.add_argument("--list-sessions", action="store_true", help="List saved sessions")
    parser.add_argument("--no-cost", action="store_true", help="Hide cost comparison")
    parser.add_argument("--no-validate", action="store_true", help="Skip the verification step")
    parser.add_argument("--output", "-o", help="Save report to file")
    args = parser.parse_args()

    config = Config()
    config.initialize()

    if args.model:
        config.model.model_name = args.model
    if args.no_cost:
        config.ui.show_cost = False

    ui = ResearchUI(config)
    ui.print_banner()

    if args.list_sessions:
        persistence = StatePersistence(config)
        sessions = persistence.list_sessions()
        if not sessions:
            ui.print_error("No saved sessions found.")
            return
        from rich.table import Table
        table = Table(title="Saved Sessions", show_header=True, header_style="bold cyan")
        table.add_column("Session ID")
        table.add_column("Query")
        table.add_column("Saved At")
        for s in sessions:
            table.add_row(s["session_id"], s["query"][:50], s["saved_at"][:19])
        from rich.console import Console
        Console().print(table)
        return

    if args.resume:
        persistence = StatePersistence(config)
        saved = persistence.load_state(args.resume)
        if saved:
            ui.print_error(f"Resuming session {args.resume} (not yet fully implemented)")
            return
        else:
            ui.print_error(f"Session {args.resume} not found.")
            return

    if not args.query:
        query = ui.prompt_query()
        if not query:
            ui.print_error("No query provided. Exiting.")
            sys.exit(1)
    else:
        query = args.query

    ui.print_cost_comparison(tokens_used=0)

    from rich.live import Live
    from rich.panel import Panel

    with Live(Panel("Initializing research agents...", style="cyan"), refresh_per_second=4) as live:
        graph = ResearchGraph(config)

    ui.show_progress("Planning", f"Breaking down: {query[:60]}...")

    result = graph.run(query)

    if result.get("report"):
        ui.print_report(result["report"])
        tokens_used = result.get("tokens_used", 5000)
        ui.print_cost_comparison(tokens_used=tokens_used)
        ui.print_session_info(result.get("session_id", ""), tokens_used)
        if args.output:
            with open(args.output, "w") as f:
                f.write(result["report"])
            ui.print_error(f"Report saved to {args.output}")
    else:
        ui.print_error("Research completed but no report was generated.")


if __name__ == "__main__":
    main()
