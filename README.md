# Research-CLI: Autonomous Researcher for Your Terminal

> **Professional-grade research on your laptop. Zero API keys. Zero costs.**

Run deep, multi-agent research using small local models (Qwen2.5-3B, Gemma3-4B) without paying OpenAI or Tavily a single cent.

## Why This Exists

Most research agents are either:
- **Cloud-only** → Expensive ($0.50+ per query with GPT-4)
- **Naive-local** → One loop that hallucinates and fails

Research-CLI uses a **multi-agent graph architecture** where each agent has one focused job. Small models excel at narrow tasks. The architecture handles the complexity, not the model.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Research Query                        │
└────────────────────────┬────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ PLANNER │  Breaks query into 5-8 sub-tasks (DAG)
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐┌──────────┐┌──────────┐
        │Researcher││Researcher││Researcher│  Parallel via ThreadPoolExecutor
        │  (Web)   ││ (Stats)  ││ (Views)  │  Each has ONE focused task
        └────┬─────┘└────┬─────┘└────┬─────┘
             │           │           │
             └───────────┼───────────┘
                         │
                    ┌────▼────┐
                    │ CRITIC  │  Audits findings: citations? specific?
                    └────┬────┘
                         │
                ┌────────┴────────┐
                │                 │
          [FAIL: retry]     [PASS: continue]
                │                 │
                ▼                 ▼
          ┌──────────┐     ┌───────────┐
          │ Researcher│    │ VALIDATOR │  Fact-checks each claim
          │  (retry)  │    │ vs source │  "Does source support this?"
          └──────────┘     └─────┬─────┘
                                 │
                            ┌────▼────┐
                            │ WRITER  │  Synthesizes final report
                            └────┬────┘
                                 │
                            ┌────▼────┐
                            │ REPORT  │  With citations & sources
                            └─────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **Zero API Cost** | Uses DuckDuckGo + Crawl4AI. No subscriptions. |
| **Self-Correction** | Validator checks every claim against its source. |
| **Crash Recovery** | State saved to JSON after every step. Resume exactly where you left off. |
| **Local RAG** | ChromaDB stores findings. Small model only sees top-3 relevant snippets. |
| **Cost Calculator** | Shows how much you saved vs GPT-4 in real-time. |
| **Beautiful CLI** | Rich-based terminal UI with progress bars and panels. |

## Installation

```bash
# Clone and install
git clone https://github.com/yourusername/research-cli.git
cd research-cli
pip install -e .

# Make sure Ollama is running with a small model
ollama pull qwen2.5:3b
ollama serve
```

## Usage

```bash
# Basic research
research "What are the latest advances in fusion energy?"

# Use a different model
research "Compare React vs Vue vs Svelte in 2025" --model llama3.1:8b

# Save report to file
research "Impact of AI on software engineering" --output report.md

# List saved sessions
research --list-sessions

# Resume a crashed session
research --resume abc123
```

### Python API

```python
from research_cli import Config, ResearchGraph

config = Config()
graph = ResearchGraph(config)
result = graph.run("What are the latest advances in fusion energy?")
print(result["report"])
```

## The Self-Correction Loop (What Makes This Different)

Every claim goes through a **Verify → Correct** cycle:

```
1. Generate: Agent writes a claim based on search results
2. Verify:  "Does the source document support this claim? YES/NO"
3. Correct: If NO → rewrite claim or trigger re-scrape
```

This turns a hallucination-prone 3B model into a fact-checker. You see every verification step in the CLI output.

## Cost Comparison

| Query | GPT-4 + Tavily | Research-CLI |
|-------|---------------|--------------|
| Simple question | $0.05 | $0.00 |
| Deep research | $0.50+ | $0.00 |
| 100 queries/mo | $50+ | $0.00 |

## Tech Stack

- **Inference**: [Ollama](https://ollama.ai) - Local model hosting
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) - Stateful agent graphs
- **Scraping**: [Crawl4AI](https://github.com/unclecode/crawl4ai) - Open-source web scraping
- **Vector Store**: [ChromaDB](https://github.com/chroma-core/chroma) - Local, file-based
- **CLI**: [Rich](https://github.com/Textualize/rich) - Beautiful terminal output

## Project Structure

```
src/research_cli/
├── main.py              # CLI entry point
├── config.py            # Centralized configuration
├── prompts/             # System prompt templates (.txt files)
│   ├── planner.txt
│   ├── researcher.txt
│   ├── critic.txt
│   ├── writer.txt
│   └── validator.txt
├── agents/
│   ├── planner.py       # DAG task generation
│   ├── researcher.py    # Parallel web research
│   ├── critic.py        # Quality audit
│   ├── validator.py     # Self-correction loop
│   └── writer.py        # Report synthesis
├── tools/
│   ├── search.py        # DuckDuckGo wrapper
│   ├── scraper.py       # Crawl4AI + urllib fallback
│   └── rag.py           # ChromaDB RAG
├── graph/
│   ├── state.py         # LangGraph state definition
│   └── workflow.py      # Full orchestration graph
├── storage/
│   └── persistence.py   # JSON crash recovery
└── ui/
    └── cli.py           # Rich terminal UI
```

## License

MIT
# Research-CLI-Autonomous-Researcher-for-Your-Terminal
