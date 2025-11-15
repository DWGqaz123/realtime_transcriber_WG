# backend/run_logger.py
from pathlib import Path
from typing import Dict, Any


class RunLogger:
    """
    Minimal stub implementation for now.

    Purpose:
    - Avoid import errors.
    - Provide simple print-based logging.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, session_id: str, meta: Dict[str, Any] | None = None) -> None:
        print(f"[RunLogger] Start run for session {session_id}, meta={meta}")

    def log_event(self, session_id: str, event: Dict[str, Any]) -> None:
        print(f"[RunLogger] Event for {session_id}: {event}")

    def finish_run(self, session_id: str) -> None:
        print(f"[RunLogger] Finish run for session {session_id}")