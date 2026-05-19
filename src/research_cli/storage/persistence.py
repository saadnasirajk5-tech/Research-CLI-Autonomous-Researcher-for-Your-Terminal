"""
State Persistence - JSON-based crash recovery.

Saves the research state to disk after each node completes.
If the process crashes, the agent resumes from the last saved state.
No variables lost, no re-doing completed work.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from research_cli.config import Config


class StatePersistence:
    """
    Saves and loads research session state to JSON files.

    Each session gets its own file named by session_id.
    State is saved after every node execution.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the persistence layer."""
        self.config = config or Config()
        self.config.initialize()

    def save_state(self, state: dict) -> str:
        """
        Save the current state to a JSON file.

        Parameters:
            state: The LangGraph state dict to persist.

        Returns:
            The session ID used for the file.
        """
        session_id = state.get("session_id", str(uuid.uuid4()))
        file_path = self.config.storage.sessions_dir / f"{session_id}.json"
        serializable = self._make_serializable(state)
        serializable["_saved_at"] = datetime.now().isoformat()
        with open(file_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        return session_id

    def load_state(self, session_id: str) -> dict | None:
        """
        Load a saved state from disk.

        Parameters:
            session_id: The session ID to load.

        Returns:
            The state dict, or None if not found.
        """
        file_path = self.config.storage.sessions_dir / f"{session_id}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)

    def list_sessions(self) -> list[dict]:
        """
        List all saved sessions.

        Returns:
            List of dicts with session_id, saved_at, query.
        """
        sessions = []
        for f in self.config.storage.sessions_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    sessions.append({
                        "session_id": f.stem,
                        "saved_at": data.get("_saved_at", ""),
                        "query": data.get("query", ""),
                    })
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s["saved_at"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session file."""
        file_path = self.config.storage.sessions_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    @staticmethod
    def _make_serializable(obj: dict) -> dict:
        """
        Convert a state dict to JSON-serializable format.

        Removes non-serializable types and converts them to strings.
        """
        result = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                result[key] = StatePersistence._make_serializable(value)
            elif isinstance(value, list):
                result[key] = [
                    StatePersistence._make_serializable(item) if isinstance(item, dict) else str(item)
                    for item in value
                ]
            elif isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            else:
                result[key] = str(value)
        return result


def save_finding_to_disk(finding: dict, config: Config | None = None) -> Path:
    """
    Save an individual finding to a JSON file for auditability.

    Each finding gets its own file so users can inspect raw data.

    Parameters:
        finding: The finding dict to save.
        config: Optional config.

    Returns:
        Path to the saved file.
    """
    config = config or Config()
    config.initialize()
    task_id = finding.get("task_id", "unknown")
    finding_id = str(uuid.uuid4())[:8]
    file_path = config.storage.findings_dir / f"{task_id}_{finding_id}.json"
    with open(file_path, "w") as f:
        json.dump(finding, f, indent=2, default=str)
    return file_path
