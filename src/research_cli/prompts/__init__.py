"""
Prompt Loader - Load system prompts from template files.

Instead of embedding long prompts in config.py, prompts are stored
as .txt files in the prompts/ directory. This makes them easier to
edit, version, and maintain as they grow.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """
    Load a prompt template by name.

    Parameters:
        name: The prompt name (without .txt extension).
              e.g., "planner", "researcher", "critic", "writer", "validator"

    Returns:
        The prompt text content.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    prompt_path = PROMPTS_DIR / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text().strip()
