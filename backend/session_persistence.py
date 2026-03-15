from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging
import time

from database.db import DatabaseManager
from summary_service import SummaryService

log = logging.getLogger("transcriber.persistence")


class SessionPersistenceService:
    """Persistence operations extracted from SessionManager."""

    def __init__(self, indexing_service: Any):
        self.indexing_service = indexing_service

    def save_session(self, session: Any) -> dict:
        duration = int(time.time() - session.start_time)
        full_transcript = "\n".join(session.transcript_parts)
        sentence_count = len(session.transcript_parts)
        char_count = len(full_transcript)

        DatabaseManager.update_session(
            session_id=session.db_session_id,
            duration_seconds=duration,
            transcript_text=full_transcript,
            sentence_count=sentence_count,
            char_count=char_count,
            ended_at=datetime.utcnow(),
        )

        session.is_saved = True

        project_id = session.meta.get("project_id")
        if project_id is not None:
            project = DatabaseManager.get_project_by_id(project_id)
            if project:
                self._save_transcript_to_file(
                    project_name=str(project.name),
                    mode=session.meta.get("mode", "unknown"),
                    transcript_text=full_transcript,
                    started_at=session.started_at.isoformat() if session.started_at else datetime.now().isoformat(),
                )

        return {
            "status": "saved",
            "session_id": session.db_session_id,
            "duration": duration,
            "sentences": sentence_count,
            "characters": char_count,
            "summaries": session.summary_count,
        }

    async def index_session_summaries(self, db_session_id: int) -> dict:
        return await self.indexing_service.index_session_summaries(db_session_id)

    async def create_final_summary(self, session: Any) -> Optional[dict]:
        full_transcript = "\n".join(session.transcript_parts)
        if not full_transcript or len(full_transcript.strip()) < 20 or not session.db_session_id:
            return None

        summary_content = await SummaryService().generate_summary(
            buffer_text=full_transcript,
            context="",
        )
        if not summary_content or not summary_content.strip():
            return None

        summary = DatabaseManager.create_summary(
            session_id=session.db_session_id,
            content=summary_content,
            source_text=full_transcript[:1000],
            start_sentence_idx=0,
            end_sentence_idx=len(session.transcript_parts),
            duration_seconds=int(time.time() - session.start_time),
        )

        return {
            "id": summary.id,
            "content": summary_content,
            "created_at": summary.created_at.isoformat() if summary.created_at else "",
            "is_final": True,
        }

    def _save_transcript_to_file(
        self,
        project_name: str,
        mode: str,
        transcript_text: str,
        started_at: str,
    ) -> Optional[str]:
        try:
            base_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "transcripts"
            safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (" ", "-", "_")).strip()
            project_dir = base_dir / safe_project_name
            project_dir.mkdir(parents=True, exist_ok=True)

            try:
                dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now()

            filename = f"{dt.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}.txt"
            filepath = project_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Project: {project_name}\n")
                f.write(f"Mode: {mode}\n")
                f.write(f"Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(transcript_text)

            log.info("Transcript saved to file: %s", filepath)
            return str(filepath)
        except Exception as exc:
            log.error("Failed to save transcript to file: %s", exc, exc_info=True)
            return None
