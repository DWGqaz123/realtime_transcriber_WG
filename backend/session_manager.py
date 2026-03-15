import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from fastapi import WebSocket

from config import TranscriptionConfig, LogConfig, OPENAI_API_KEY
from database.db import DatabaseManager
from elevenlabs_client import ElevenLabsRealtimeClient, ElevenLabsConfig
from indexing_service import get_indexing_service
from live_session import LiveSession
from session_persistence import SessionPersistenceService
from summary_handler import SummaryHandler
from summary_service import SummaryService

log = logging.getLogger("transcriber.session")


class SessionManager:
    """
    Coordinates all transcription sessions.

    Responsibilities:
    - Session lifecycle (create / close).
    - Route incoming WebSocket messages to command handlers.
    - Manage ElevenLabs connection and manual-commit loop.
    - Delegate summary work to SummaryHandler.
    """

    def __init__(self, api_key: Optional[str] = None):
        log.info("SessionManager initializing...")

        self.sessions: Dict[str, LiveSession] = {}
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        if not self._api_key:
            log.warning("No ElevenLabs API key available!")

        self.indexing_service = get_indexing_service()

        try:
            self.summary_service = SummaryService()
        except Exception as e:
            log.error("Failed to initialize SummaryService: %s", e)
            self.summary_service = None

        self.persistence = SessionPersistenceService(self.indexing_service, self.summary_service)
        self.summary_handler = SummaryHandler(self.summary_service, self.persistence)

        log.info(
            "SessionManager ready: elevenlabs=%s, openai=%s, summary=%s",
            "ok" if self._api_key else "missing",
            "ok" if OPENAI_API_KEY else "missing",
            "ok" if self.summary_service else "missing",
        )

    # ── WebSocket helpers ────────────────────────────────────────────────────

    async def _send_event(
        self, websocket: WebSocket, event_type: str, payload: Optional[dict] = None
    ) -> None:
        await websocket.send_text(json.dumps({"type": event_type, "payload": payload or {}}))

    async def _send_session_event(
        self, session: LiveSession, event_type: str, payload: Optional[dict] = None
    ) -> None:
        await self._send_event(session.websocket, event_type, payload)

    def _make_send_fn(self, session: LiveSession):
        """Return an async callable send_fn(event_type, payload) bound to a session."""
        async def send_fn(event_type: str, payload: Optional[dict] = None):
            await self._send_session_event(session, event_type, payload)
        return send_fn

    # ── Session lifecycle ────────────────────────────────────────────────────

    async def create_session(self, websocket: WebSocket) -> LiveSession:
        session_id = str(uuid4())
        session = LiveSession(
            id=session_id, websocket=websocket, meta={}, started_at=datetime.now()
        )
        self.sessions[session_id] = session
        log.info("Session created: %s", session_id)
        return session

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if not session:
            log.warning("Session %s not found", session_id)
            return

        log.info("Closing session %s", session_id)
        session.is_active = False

        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            try:
                await session.manual_commit_task
            except asyncio.CancelledError:
                pass

        if session.eleven_client is not None:
            try:
                await session.eleven_client.close()
            except Exception as exc:
                log.warning("Error closing ElevenLabs client: %s", exc)

        try:
            from fastapi.websockets import WebSocketState
            if (
                not hasattr(session.websocket, "client_state")
                or session.websocket.client_state != WebSocketState.DISCONNECTED
            ):
                await session.websocket.close(code=1000, reason="Session closed normally")
        except Exception as exc:
            log.debug("WebSocket already closed: %s", exc)

        del self.sessions[session_id]
        log.info("Session %s cleaned up", session_id)

    # ── ElevenLabs connection management ─────────────────────────────────────

    def _build_elevenlabs_config(self, mode: str) -> ElevenLabsConfig:
        mc = TranscriptionConfig.get_mode_config(mode)
        return ElevenLabsConfig(
            audio_format=mc.audio_format,
            sample_rate=mc.sample_rate,
            language_code=mc.language_code,
            timestamps_granularity=mc.timestamps_granularity,
            model_id=mc.model_id,
            mode=mode,
            commit_strategy=mc.commit_strategy,
            vad_silence_threshold_secs=mc.vad_silence_threshold_secs,
            vad_threshold=mc.vad_threshold,
            min_speech_duration_ms=mc.min_speech_duration_ms,
            min_silence_duration_ms=mc.min_silence_duration_ms,
        )

    async def _start_manual_commit_loop(self, session: LiveSession, mode: str) -> None:
        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            try:
                await session.manual_commit_task
            except asyncio.CancelledError:
                pass
        commit_interval = TranscriptionConfig.get_mode_config(mode).commit_interval
        assert commit_interval is not None
        session.manual_commit_task = asyncio.create_task(
            self._manual_commit_loop(session.id, commit_interval)
        )

    async def _ensure_elevenlabs_connection(self, session: LiveSession) -> bool:
        if session.eleven_client is None:
            log.warning("ElevenLabs client not initialized for %s", session.id)
            return False

        if session.eleven_client.is_alive():
            return True

        log.warning("ElevenLabs connection dead for %s, reconnecting...", session.id)
        try:
            await session.eleven_client.close()
            await session.eleven_client.connect()
            log.info("Reconnected to ElevenLabs for %s", session.id)
            mode = session.meta.get("mode", "lecture")
            if TranscriptionConfig.get_mode_config(mode).commit_strategy == "manual":
                await self._start_manual_commit_loop(session, mode)
            return True
        except Exception as exc:
            error_msg = f"Failed to reconnect to ElevenLabs: {exc}"
            log.error(error_msg)
            try:
                await self._send_session_event(session, "error", {"message": error_msg})
            except Exception:
                pass
            return False

    async def _manual_commit_loop(self, session_id: str, interval: float) -> None:
        log.debug("Manual commit loop started for %s (interval=%ds)", session_id, interval)
        try:
            while True:
                await asyncio.sleep(interval)
                session = self.sessions.get(session_id)
                if session is None or not session.is_active:
                    break
                if session.eleven_client is not None and session.eleven_client.is_alive():
                    try:
                        await session.eleven_client.send_commit()
                        log.debug("Manual commit sent for %s", session_id)
                    except Exception as exc:
                        log.error("Error sending manual commit: %s", exc)
        except asyncio.CancelledError:
            log.debug("Manual commit loop cancelled for %s", session_id)
            raise
        except Exception as exc:
            log.error("Unexpected error in manual commit loop: %s", exc)

    # ── Database helpers ─────────────────────────────────────────────────────

    def _reset_summary_state(self, session: LiveSession) -> None:
        session.last_summary_time = time.time()
        session.ingestion_buffer.clear()
        session.context_cache.clear()
        session.is_generating = False
        session.summary_count = 0

    def _prepare_database_session(self, session: LiveSession, mode: str) -> None:
        project_id = session.meta.get("project_id")
        if project_id is None:
            log.warning("No project_id for session %s", session.id)
            return
        try:
            if session.db_session_id is not None and not session.is_saved:
                log.info("Reusing existing DB session: %s", session.db_session_id)
                self._reset_summary_state(session)
                return
            db_session = DatabaseManager.create_session(project_id=project_id, mode=mode)
            session.db_session_id = db_session.id
            session.is_saved = False
            self._reset_summary_state(session)
            session.transcript_parts.clear()
            log.info(
                "Created new DB session: %d (project=%d, mode=%s)",
                db_session.id, project_id, mode,
            )
        except Exception as exc:
            log.error("Failed to create/reuse DB session: %s", exc, exc_info=True)

    async def _save_session_to_db(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None or session.is_saved:
            return
        if session.db_session_id is None:
            await self._send_save_error(session, "No database session")
            return
        if not session.transcript_parts:
            await self._send_save_error(session, "No transcript to save")
            return
        try:
            save_result = self.persistence.save_session(session)
            log.info("Saved session to DB: id=%s", session.db_session_id)
            await self._send_session_event(session, "save_complete", save_result)
            if session.summary_count > 0:
                try:
                    await self._send_session_event(session, "indexing_start")
                except Exception:
                    pass
                asyncio.create_task(
                    self._index_session_async(session.db_session_id, session.websocket)
                )
            else:
                log.debug("No summaries to index")
        except Exception as exc:
            log.error("Failed to save session: %s", exc, exc_info=True)
            await self._send_save_error(session, str(exc))

    async def _send_save_error(self, session: LiveSession, message: str) -> None:
        log.warning("Save error for %s: %s", session.id, message)
        try:
            await self._send_session_event(
                session, "save_status", {"status": "error", "message": message}
            )
        except Exception:
            pass

    async def _index_session_async(self, db_session_id: int, websocket: WebSocket) -> None:
        try:
            log.info("Starting async indexing for session %s", db_session_id)
            result = await self.persistence.index_session_summaries(db_session_id)
            if result["success"]:
                try:
                    await self._send_event(websocket, "indexing_complete", {
                        "status": "completed",
                        "session_id": db_session_id,
                        "indexed_count": result.get("indexed", 0),
                        "project_id": result.get("project_id"),
                    })
                    log.info("Indexing complete: %d summaries", result.get("indexed", 0))
                except Exception as e:
                    log.warning("Failed to send indexing_complete: %s", e)
            else:
                log.error("Indexing failed for session %d: %s", db_session_id, result.get("error"))
                try:
                    await self._send_event(websocket, "indexing_error", {
                        "status": "failed",
                        "session_id": db_session_id,
                        "error": result.get("error", "Unknown error"),
                    })
                except Exception as e:
                    log.warning("Failed to send indexing_error: %s", e)
        except Exception as e:
            log.error("Async indexing failed: %s", e, exc_info=True)
            try:
                await self._send_event(websocket, "indexing_error", {
                    "status": "failed",
                    "session_id": db_session_id,
                    "error": str(e),
                })
            except Exception:
                pass

    # ── Command handlers ─────────────────────────────────────────────────────

    async def _handle_project_command(self, session: LiveSession, text: str) -> None:
        try:
            project_id = int(text.split(":", 1)[1].strip())
        except (ValueError, IndexError):
            await self._send_session_event(session, "error", {"message": "Invalid project_id"})
            return
        if not DatabaseManager.get_project_by_id(project_id):
            log.warning("Project %d not found", project_id)
            await self._send_session_event(session, "error", {"message": "Project not found"})
            return
        session.meta["project_id"] = project_id
        log.info("Set project_id=%d for session %s", project_id, session.id)
        await self._send_session_event(session, "config", {"project_id": project_id})

    async def _handle_mode_command(self, session: LiveSession, text: str) -> None:
        mode_raw = text.split(":", 1)[1].strip().lower()
        mode = "discussion" if mode_raw == "discussion" else "lecture"
        session.meta["mode"] = mode
        log.info("MODE command: session=%s, mode=%s", session.id, mode)

        if session.eleven_client is not None:
            await self._send_session_event(session, "config", {"mode": mode, "status": "already_set"})
            return

        try:
            config = self._build_elevenlabs_config(mode)
            eleven_client = ElevenLabsRealtimeClient(api_key=self._api_key, config=config)
            eleven_client.on_partial = lambda t: asyncio.create_task(
                self.push_transcript_to_client(session.id, t, is_final=False)
            )
            eleven_client.on_final = lambda t: asyncio.create_task(
                self.push_transcript_to_client(session.id, t, is_final=True)
            )
            await eleven_client.connect()
            session.eleven_client = eleven_client
            log.info("Connected to ElevenLabs: session=%s, mode=%s", session.id, mode)

            self._prepare_database_session(session, mode)

            if config.commit_strategy == "manual":
                await self._start_manual_commit_loop(session, mode)
                log.info("Started manual commit loop")

        except Exception as exc:
            error_msg = f"Failed to connect to ElevenLabs: {exc}"
            log.error(error_msg)
            session.meta["connection_error"] = str(exc)
            await self._send_session_event(session, "error", {"message": error_msg})

    async def _handle_stop_command(self, session: LiveSession) -> None:
        log.info("Received STOP signal from %s", session.id)
        full_transcript = "\n".join(session.transcript_parts)

        if full_transcript.strip() and len(full_transcript.strip()) > 20 and session.db_session_id:
            try:
                await self._send_session_event(session, "summary_started", {"is_final": True})
            except Exception:
                pass
            try:
                await self.summary_handler.generate_final(session, self._make_send_fn(session))
            except Exception as exc:
                log.error("Failed to generate final summary: %s", exc, exc_info=True)
        else:
            log.debug("No final summary: transcript too short or no db session")

        # Auto-save transcript so it's persisted even if user never clicks Save
        if not session.is_saved:
            await self._save_session_to_db(session.id)

        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            session.manual_commit_task = None

        if session.eleven_client is not None:
            await session.eleven_client.close()
            session.eleven_client = None

        try:
            await self._send_session_event(
                session, "status", {"message": "recording stopped, session kept alive"}
            )
        except Exception as exc:
            log.warning("Failed to send stop acknowledgment: %s", exc)

    # ── Public message handlers ───────────────────────────────────────────────

    async def handle_binary_audio(self, session_id: str, data: bytes) -> None:
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            return
        if not await self._ensure_elevenlabs_connection(session):
            return
        try:
            await session.eleven_client.send_audio_chunk(data)
        except Exception as exc:
            log.error("Failed to send audio: %s", exc)
            try:
                await self._send_session_event(session, "error", {"message": str(exc)})
            except Exception:
                pass

    async def handle_text_message(self, session_id: str, text: str) -> None:
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            return

        stripped = text.strip()
        upper = stripped.upper()

        if upper.startswith("PROJECT_ID:"):
            await self._handle_project_command(session, stripped)
        elif upper.startswith("MODE:"):
            await self._handle_mode_command(session, stripped)
        elif upper == "STOP":
            await self._handle_stop_command(session)
        elif upper == "SAVE":
            log.info("Received SAVE for session %s", session_id)
            await self._save_session_to_db(session_id)
        else:
            log.debug("Text from %s: %s", session_id, text[:100])
            try:
                await self._send_session_event(session, "echo", {"text": text})
            except Exception as exc:
                log.warning("Failed to send echo: %s", exc)

    async def push_transcript_to_client(
        self, session_id: str, text: str, is_final: bool
    ) -> None:
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            return

        if is_final and text.strip():
            stripped = text.strip()
            session.transcript_parts.append(stripped)
            session.ingestion_buffer.append(stripped)

            if self.summary_handler.should_generate(session):
                if LogConfig.LOG_SUMMARY:
                    log.info("Summary trigger met, initiating generation")
                try:
                    await self._send_session_event(session, "summary_started", {"is_final": False})
                except Exception:
                    pass
                asyncio.create_task(
                    self.summary_handler.generate_and_send(session, self._make_send_fn(session))
                )

        msg_type = "final" if is_final else "partial"
        try:
            await self._send_session_event(session, msg_type, {"text": text})
        except Exception as exc:
            log.error("Failed to push transcript to %s: %s", session_id, exc, exc_info=True)
