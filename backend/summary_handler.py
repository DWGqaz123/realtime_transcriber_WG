import time
import logging
from datetime import datetime
from typing import Any, Callable, Awaitable, List

from config import SummaryConfig
from database.db import DatabaseManager

log = logging.getLogger("transcriber.summary")

# Type alias: async fn(event_type, payload) -> None
SendFn = Callable[[str, dict], Awaitable[None]]


class SummaryHandler:
    """Encapsulates all summary triggering, generation, and persistence logic."""

    def __init__(self, summary_service: Any, persistence: Any):
        self.summary_service = summary_service
        self.persistence = persistence

    # ── Trigger check ────────────────────────────────────────────────────────

    def should_generate(self, session: Any) -> bool:
        elapsed = time.time() - session.last_summary_time

        if not (
            elapsed >= SummaryConfig.SUMMARY_INTERVAL_SECONDS
            and len(session.ingestion_buffer) >= SummaryConfig.MIN_SENTENCES
            and session.is_active
            and session.db_session_id is not None
        ):
            return False

        if elapsed >= SummaryConfig.LOOSE_MODE_THRESHOLD:
            log.debug("Loose mode trigger: %.1fs", elapsed)
            return True

        if session.ingestion_buffer and self._has_sentence_ender(session.ingestion_buffer[-1]):
            log.debug("Semantic check passed")
            return True

        log.debug("Waiting for sentence completion")
        return False

    def _has_sentence_ender(self, text: str) -> bool:
        text = text.strip()
        return bool(text) and any(text.endswith(e) for e in SummaryConfig.SENTENCE_ENDERS)

    # ── Incremental summary ──────────────────────────────────────────────────

    async def generate_and_send(self, session: Any, send_fn: SendFn) -> None:
        log.info("Summary start: session=%s, db=%s", session.id, session.db_session_id)

        snapshot = session.ingestion_buffer.copy()
        session.ingestion_buffer.clear()
        session.is_generating = True

        if not snapshot:
            log.warning("Empty snapshot, aborting")
            session.is_generating = False
            return

        current_text = "\n".join(snapshot)
        context = "\n".join(session.context_cache)

        try:
            summary_content = await self.summary_service.generate_summary(
                buffer_text=current_text, context=context
            )
            if not summary_content:
                log.warning("Empty summary from OpenAI")
                self._cleanup(session)
                return
            log.info("Summary generated: %d chars", len(summary_content))
        except Exception as e:
            log.error("Summary generation failed: %s", e, exc_info=True)
            self._cleanup(session)
            return

        if session.db_session_id:
            try:
                duration = int(time.time() - session.last_summary_time)
                start_idx = session.next_start_sentence_idx
                summary = DatabaseManager.create_summary(
                    session_id=session.db_session_id,
                    content=summary_content,
                    source_text=current_text[:500],
                    start_sentence_idx=start_idx,
                    end_sentence_idx=start_idx + len(snapshot),
                    duration_seconds=duration,
                )
                session.summary_count += 1
                log.info("[DB] Created summary id=%d", summary.id)
                try:
                    await send_fn("summary", {
                        "id": summary.id,
                        "content": summary_content,
                        "created_at": summary.created_at.isoformat() if summary.created_at else datetime.now().isoformat(),
                        "is_final": False,
                        "sentence_count": len(snapshot),
                        "duration": duration,
                    })
                except Exception as e:
                    log.warning("Failed to send summary: %s", e)
            except Exception as e:
                log.error("Database save failed: %s", e, exc_info=True)

        session.context_cache = snapshot[-SummaryConfig.CONTEXT_CACHE_SIZE:]
        self._cleanup(session)
        log.info("Summary complete for session %s", session.id)

    # ── Final summary (on STOP) ──────────────────────────────────────────────

    async def generate_final(self, session: Any, send_fn: SendFn) -> None:
        try:
            summary_data = await self.persistence.create_final_summary(session)
            if summary_data is None:
                log.warning("Final summary skipped (short transcript or no db session)")
                return
            log.info("Final summary generated: %d chars", len(summary_data["content"]))
            await send_fn("summary", summary_data)
        except Exception as e:
            log.error("Final summary failed: %s", e, exc_info=True)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _cleanup(self, session: Any) -> None:
        session.is_generating = False
        session.last_summary_time = time.time()
