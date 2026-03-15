# backend/session_manager.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from fastapi import WebSocket
from uuid import uuid4
from config import TranscriptionConfig, SummaryConfig, LogConfig
from summary_service import SummaryService
from elevenlabs_client import ElevenLabsRealtimeClient, ElevenLabsConfig
import time
import json
import logging
from database.db import DatabaseManager
from datetime import datetime
from indexing_service import get_indexing_service
from session_persistence import SessionPersistenceService
import os

log = logging.getLogger("transcriber.session")


@dataclass
class LiveSession:
    """会话数据类 - 管理单个录音会话的状态"""
    
    id: str
    websocket: WebSocket
    meta: dict
    started_at: datetime
    
    # statistics
    start_time: float = field(default_factory=lambda: time.time())
    
    # ==================== 转录相关 ====================
    transcript_parts: List[str] = field(default_factory=list)
    
    # ==================== 摘要缓冲 ====================
    ingestion_buffer: List[str] = field(default_factory=list)
    context_cache: List[str] = field(default_factory=list)
    
    # ==================== 状态管理 ====================
    is_generating: bool = False
    last_summary_time: float = field(default_factory=lambda: time.time())
    summary_count: int = 0
    next_start_sentence_idx: int = 0
    
    # ==================== 会话状态 ====================
    is_active: bool = True
    is_saved: bool = False
    db_session_id: Optional[int] = None
    
    # ==================== ElevenLabs 相关 ====================
    # 🔧 添加缺失的字段
    eleven_client: Optional[ElevenLabsRealtimeClient] = None  # ElevenLabs 客户端实例
    manual_commit_task: Optional[asyncio.Task] = None

    
    def __post_init__(self):
        """初始化后的设置"""
        log.debug("Created session: %s", self.id)


class SessionManager:
    """
    Manages all transcription sessions.

    Responsibilities:
    - Create and close Session objects.
    - Route incoming messages (text / binary) to the correct handlers.
    - Push transcription results back to the frontend.
    - Cooperate with RunLogger to record session information.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SessionManager.

        Args:
            api_key: ElevenLabs API key. If None, read from environment.
        """

        log.info("SessionManager initializing...")

        self.sessions: Dict[str, LiveSession] = {}
        
        # 从多个来源尝试获取 API key
        env_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._api_key = api_key or env_key
        
        log.info("API Key: %s (source: %s)",
                 "configured" if self._api_key else "missing",
                 "constructor" if api_key else ("env" if env_key else "none"))
        
        if not self._api_key:
            log.warning("No ElevenLabs API key available!")
        
        # 初始化索引服务
        self.indexing_service = get_indexing_service()
        self.persistence = SessionPersistenceService(self.indexing_service)
        log.debug("IndexingService initialized")
        
        # 初始化摘要服务
        try:
            self.summary_service = SummaryService()
            log.debug("SummaryService initialized")
        except Exception as e:
            log.error("Failed to initialize SummaryService: %s", e)
            self.summary_service = None
        
        # 打印最终状态
        from config import OPENAI_API_KEY
        log.info("SessionManager ready: elevenlabs=%s, openai=%s, summary=%s, indexing=ok",
                 "ok" if self._api_key else "missing",
                 "ok" if OPENAI_API_KEY else "missing",
                 "ok" if self.summary_service else "missing")

    async def _send_event(self, websocket: WebSocket, event_type: str, payload: Optional[dict] = None) -> None:
        message = {
            "type": event_type,
            "payload": payload or {},
        }
        await websocket.send_text(json.dumps(message))

    async def _send_session_event(self, session: LiveSession, event_type: str, payload: Optional[dict] = None) -> None:
        await self._send_event(session.websocket, event_type, payload)
    
    def _has_sentence_ender(self, text: str) -> bool:
        """
        检查文本是否以句子结束符结尾
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: 是否以结束符结尾
        """
        if not text:
            return False
        
        text = text.strip()
        
        # 检查是否以任意结束符结尾
        for ender in SummaryConfig.SENTENCE_ENDERS:
            if text.endswith(ender):
                return True
        
        return False
    
    def _update_context_cache(self, session: LiveSession, sentences: List[str]) -> None:
        """
        更新上下文缓存（保留最后 N 句）
        
        Args:
            session: 会话对象
            sentences: 新处理的句子列表
        """
        if not sentences:
            return
        
        # 取最后 CONTEXT_CACHE_SIZE 句
        cache_size = SummaryConfig.CONTEXT_CACHE_SIZE
        session.context_cache = sentences[-cache_size:]
        
        log.debug("Updated context cache: %d sentences", len(session.context_cache))
    
    async def _generate_and_send_summary(self, session: LiveSession) -> None:
        """Generate and persist an incremental summary for the session."""
        session_id = session.id
        
        log.info("Summary generation start: session=%s, db_session=%s", session_id, session.db_session_id)
        start_time = time.time()
        
        snapshot = session.ingestion_buffer.copy()
        session.ingestion_buffer.clear()
        session.is_generating = True
        
        elapsed = (time.time() - start_time) * 1000
        log.debug("Atomic transfer: %d sentences, %.3fms", len(snapshot), elapsed)
        
        if not snapshot:
            log.warning("Empty snapshot, aborting summary generation")
            session.is_generating = False
            return

        context = "\n".join(session.context_cache)
        current_text = "\n".join(snapshot)
        
        log.debug("Summary input: context=%d chars, current=%d chars", len(context), len(current_text))

        try:
            summary_content = await self.summary_service.generate_summary(
                buffer_text=current_text,
                context=context,
            )
            
            if not summary_content:
                log.warning("Empty summary from OpenAI, aborting")
                self._cleanup_after_summary(session, snapshot)
                return
            
            log.info("Summary generated: %d chars", len(summary_content))
            
        except Exception as e:
            log.error("Summary generation failed: %s", e, exc_info=True)
            self._cleanup_after_summary(session, snapshot)
            return

        if session.db_session_id:
            try:
                from database.db import DatabaseManager
                
                project_id = session.meta.get("project_id")
                start_idx = session.next_start_sentence_idx
                end_idx = start_idx + len(snapshot)
                
                duration = int(time.time() - session.last_summary_time)
                source_text = current_text[:500]  # 保存前500字符作为来源
                
                summary = DatabaseManager.create_summary(
                    session_id=session.db_session_id,
                    content=summary_content,
                    source_text=source_text,
                    start_sentence_idx=start_idx,
                    end_sentence_idx=end_idx,
                    duration_seconds=duration
                )
                
                session.summary_count += 1
                
                log.info("[DB] Created summary: id=%d, session=%s", summary.id, session.db_session_id)
                try:
                    summary_data = {
                        "id": summary.id,
                        "content": summary_content,
                        "created_at": summary.created_at.isoformat() if summary.created_at else datetime.now().isoformat(),
                        "is_final": False,
                        "sentence_count": len(snapshot),
                        "duration": duration
                    }
                    
                    await self._send_session_event(session, "summary", summary_data)
                    
                    log.debug("Sent summary to client: id=%d", summary_data['id'])
                except Exception as e:
                    log.warning("Failed to send summary to client: %s", e)
            except Exception as e:
                log.error("Database save failed: %s", e, exc_info=True)

        self._update_context_cache(session, snapshot)
        self._cleanup_after_summary(session, snapshot)
        
        log.info("Summary generation complete for session %s", session_id)
        
    def _build_elevenlabs_config_for_mode(self, mode: str) -> ElevenLabsConfig:
        """
        Build an ElevenLabsConfig for the given mode.
        
        Now uses centralized configuration from config.py.
        
        Args:
            mode: "lecture" or "discussion"
        
        Returns:
            ElevenLabsConfig with mode-specific settings
        """
        # use centralized config
        mode_config = TranscriptionConfig.get_mode_config(mode)
        
        return ElevenLabsConfig(
            # Common settings
            audio_format=mode_config.audio_format,
            sample_rate=mode_config.sample_rate,
            language_code=mode_config.language_code,
            timestamps_granularity=mode_config.timestamps_granularity,
            model_id=mode_config.model_id,
            mode=mode,
            
            # Commit strategy
            commit_strategy=mode_config.commit_strategy,
            
            # VAD settings (None for manual mode)
            vad_silence_threshold_secs=mode_config.vad_silence_threshold_secs,
            vad_threshold=mode_config.vad_threshold,
            min_speech_duration_ms=mode_config.min_speech_duration_ms,
            min_silence_duration_ms=mode_config.min_silence_duration_ms,
        )

    async def create_session(self, websocket: WebSocket) -> LiveSession:
        """
        Create a new session for the given WebSocket connection.

        - Creates the Session object with basic metadata.
        - ElevenLabs client will be created lazily when MODE is received.
        - This allows users to connect without immediately consuming resources.
        """
        session_id = str(uuid4())
        session = LiveSession(
            id=session_id,
            websocket=websocket,
            meta={},  
            started_at=datetime.now()  
        )
        self.sessions[session_id] = session
        log.info("Session created: %s", session_id)
        return session

    async def close_session(self, session_id: str) -> None:
        """
        Close the session with the given ID.
        """
        session = self.sessions.get(session_id)
        if not session:
            log.warning("Session %s not found, already closed?", session_id)
            return

        log.info("Closing session %s", session_id)
    
        
        session.is_active = False

        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            try:
                await session.manual_commit_task
            except asyncio.CancelledError:
                pass
            log.debug("Cancelled manual commit task for %s", session_id)

        if session.eleven_client is not None:
            try:
                await session.eleven_client.close()
                log.debug("Closed ElevenLabs client for %s", session_id)
            except Exception as exc:
                log.warning("Error closing ElevenLabs client: %s", exc)

        try:
            if hasattr(session.websocket, 'client_state'):
                from fastapi.websockets import WebSocketState
                if session.websocket.client_state != WebSocketState.DISCONNECTED:
                    await session.websocket.close(code=1000, reason="Session closed normally")
            else:
                await session.websocket.close()
        except Exception as exc:
            log.debug("WebSocket already closed or error: %s", exc)

        del self.sessions[session_id]
        log.info("Session %s fully cleaned up", session_id)
    
    def _should_generate_summary(self, session: LiveSession) -> bool:
        """
        检查是否应该生成摘要
        
        条件：
        1. 距离上次摘要已超过配置的时间间隔
        2. 累积了足够数量的句子
        3. 会话仍然活跃
        4. 有关联的数据库会话ID
        5. (可选) 语义完整性检查
        """
        # 🔧 使用配置类
        interval = SummaryConfig.SUMMARY_INTERVAL_SECONDS
        min_sentences = SummaryConfig.MIN_SENTENCES
        loose_threshold = SummaryConfig.LOOSE_MODE_THRESHOLD
        
        now = time.time()
        elapsed = now - session.last_summary_time
        
        log.debug("Trigger check: time=%.1f/%ds, buffer=%d/%d, active=%s, db=%s, generating=%s",
                  elapsed, interval, len(session.ingestion_buffer), min_sentences,
                  session.is_active, session.db_session_id is not None, session.is_generating)
        
        # 基础条件检查
        has_time = elapsed >= interval
        has_sentences = len(session.ingestion_buffer) >= min_sentences
        is_active = session.is_active
        has_db_session = session.db_session_id is not None
        
        if not (has_time and has_sentences and is_active and has_db_session):
            return False
        
        # 如果超过宽松模式阈值，忽略语义检查
        if elapsed >= loose_threshold:
            log.debug("Loose mode trigger: %.1fs >= %ds", elapsed, loose_threshold)
            return True
        
        # 标准模式：检查最后一句是否完整
        if session.ingestion_buffer:
            last_sentence = session.ingestion_buffer[-1]
            has_ender = self._has_sentence_ender(last_sentence)
            
            if has_ender:
                log.debug("Semantic check passed: sentence complete")
                return True
            else:
                log.debug("Waiting for sentence completion")
                return False
        
        return False

    async def _send_save_error(self, session: LiveSession, message: str) -> None:
        log.warning("Save rejected for %s: %s", session.id, message)
        try:
            await self._send_session_event(session, "save_status", {
                "status": "error",
                "message": message,
            })
        except Exception:
            pass
              
    async def _save_session_to_db(self, session_id: str) -> None:
        """
        save the current session's transcript to the database.
        """
        session = self.sessions.get(session_id)
        if session is None:
            log.warning("Cannot save: session %s not found", session_id)
            return
        
        if session.is_saved:
            log.debug("Session %s already saved", session_id)
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

            if session.db_session_id and session.summary_count > 0:
                log.info("Triggering indexing for session %s", session.db_session_id)
                
                try:
                    await self._send_session_event(session, "indexing_start")
                except Exception as e:
                    log.warning("Failed to send indexing_start: %s", e)
                
                # 异步触发索引任务（不阻塞保存操作）
                asyncio.create_task(
                    self._index_session_async(session.db_session_id, session.websocket)
                )
            else:
                if session.summary_count == 0:
                    log.debug("No summaries to index")
                else:
                    log.warning("Cannot index: no db_session_id")
            
        except Exception as exc:
            log.error("Failed to save session to DB: %s", exc, exc_info=True)
            await self._send_save_error(session, str(exc))

    def _reset_summary_state(self, session: LiveSession) -> None:
        session.last_summary_time = time.time()
        session.ingestion_buffer.clear()
        session.context_cache.clear()
        session.is_generating = False
        session.summary_count = 0

    async def _handle_project_command(self, session: LiveSession, session_id: str, text: str) -> None:
        try:
            project_id = int(text.split(":", 1)[1].strip())
        except (ValueError, IndexError) as exc:
            log.warning("Invalid PROJECT_ID message: %s, error: %s", text, exc)
            await self._send_session_event(session, "error", {"message": "Invalid project_id"})
            return

        project = DatabaseManager.get_project_by_id(project_id)
        if not project:
            log.warning("Project %d not found", project_id)
            await self._send_session_event(session, "error", {"message": "Project not found"})
            return

        session.meta["project_id"] = project_id
        log.info("Set project_id=%d for session %s", project_id, session_id)
        await self._send_session_event(session, "config", {"project_id": project_id})

    def _prepare_database_session(self, session: LiveSession, mode: str) -> None:
        project_id = session.meta.get("project_id")
        if project_id is None:
            log.warning("No project_id in session.meta for %s", session.id)
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
            log.info("Created new DB session: %d for project %d (mode=%s)",
                     db_session.id, project_id, mode)
        except Exception as exc:
            log.error("Failed to create/reuse DB session: %s", exc, exc_info=True)

    async def _handle_mode_command(self, session: LiveSession, session_id: str, text: str) -> None:
        mode_raw = text.split(":", 1)[1].strip().lower()
        mode = "discussion" if mode_raw == "discussion" else "lecture"

        log.info("MODE command: session=%s, mode=%s, project_id=%s",
                 session_id, mode, session.meta.get("project_id"))
        session.meta["mode"] = mode

        if session.eleven_client is not None:
            log.debug("Mode already configured for %s, skipping", session_id)
            await self._send_session_event(session, "config", {"mode": mode, "status": "already_set"})
            return

        log.info("Setting mode '%s' for session %s", mode, session_id)
        try:
            config = self._build_elevenlabs_config_for_mode(mode)
            eleven_client = ElevenLabsRealtimeClient(api_key=self._api_key, config=config)

            def on_partial(text: str):
                asyncio.create_task(self.push_transcript_to_client(session_id, text, is_final=False))

            def on_final(text: str):
                asyncio.create_task(self.push_transcript_to_client(session_id, text, is_final=True))

            eleven_client.on_partial = on_partial
            eleven_client.on_final = on_final

            await eleven_client.connect()
            session.eleven_client = eleven_client
            log.info("Connected to ElevenLabs for session %s (mode=%s)", session_id, mode)

            self._prepare_database_session(session, mode)

            if config.commit_strategy == "manual":
                commit_interval = TranscriptionConfig.get_mode_config(mode).commit_interval
                assert commit_interval is not None
                session.manual_commit_task = asyncio.create_task(
                    self._manual_commit_loop(session_id, commit_interval)
                )
                log.info("Started manual commit loop: interval=%ds", commit_interval)

        except Exception as exc:
            error_msg = f"Failed to connect to ElevenLabs: {str(exc)}"
            log.error("[error] %s", error_msg)
            session.meta["connection_error"] = str(exc)
            await self._send_session_event(session, "error", {"message": error_msg})

    async def _handle_stop_command(self, session: LiveSession, session_id: str) -> None:
        log.info("Received STOP signal from %s", session_id)
        full_transcript = "\n".join(session.transcript_parts)

        if full_transcript and len(full_transcript.strip()) > 20 and session.db_session_id is not None:
            log.info("Generating final summary: %d sentences, %d chars",
                     len(session.transcript_parts), len(full_transcript))
            try:
                await self._send_session_event(session, "summary_started", {"is_final": True})
            except Exception as exc:
                log.warning("Failed to send final_summary_start: %s", exc)

            try:
                await self._generate_final_summary(session)
            except Exception as exc:
                log.error("Failed to generate final summary: %s", exc, exc_info=True)
        elif len(full_transcript.strip()) <= 20:
            log.debug("Transcript too short for final summary (%d chars)", len(full_transcript))
        else:
            log.debug("No database session, skipping final summary")

        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            session.manual_commit_task = None
            log.debug("Cancelled manual commit task for %s", session_id)

        if session.eleven_client is not None:
            await session.eleven_client.close()
            session.eleven_client = None
            log.debug("Closed ElevenLabs client for %s", session_id)

        try:
            await self._send_session_event(session, "status", {
                "message": "recording stopped, session kept alive",
            })
        except Exception as exc:
            log.warning("Failed to send stop acknowledgment: %s", exc)
    
    async def _index_session_async(self, db_session_id: int, websocket: WebSocket):
        """
        异步索引会话摘要
        
        Args:
            db_session_id: 数据库 session ID
            websocket: WebSocket 连接（用于发送完成通知）
        """
        try:
            log.info("Starting async indexing for session %s", db_session_id)
            
            result = await self.persistence.index_session_summaries(db_session_id)
            
            if result["success"]:
                try:
                    await self._send_event(websocket, "indexing_complete", {
                        "status": "completed",
                        "session_id": db_session_id,
                        "indexed_count": result.get("indexed", 0),
                        "project_id": result.get("project_id")
                    })
                    log.info("Indexing complete: %d summaries", result.get("indexed", 0))
                except Exception as e:
                    log.warning("Failed to send indexing_complete: %s", e)
            else:
                try:
                    await self._send_event(websocket, "indexing_error", {
                        "status": "failed",
                        "session_id": db_session_id,
                        "error": result.get("error", "Unknown error")
                    })
                    log.error("Indexing failed: %s", result.get("error"))
                except Exception as e:
                    log.warning("Failed to send indexing_error: %s", e)

        except Exception as e:
            log.error("Async indexing failed: %s", e, exc_info=True)
            try:
                await self._send_event(websocket, "indexing_error", {
                    "status": "failed",
                    "session_id": db_session_id,
                    "error": str(e)
                })
            except Exception:
                pass

    async def _ensure_elevenlabs_connection(self, session: LiveSession, session_id: str) -> bool:
        if session.eleven_client is None:
            log.warning("Audio received but ElevenLabs client not initialized for %s", session_id)
            return False

        if session.eleven_client.is_alive():
            return True

        log.warning("ElevenLabs connection dead for %s, reconnecting...", session_id)

        try:
            await session.eleven_client.close()
            await session.eleven_client.connect()
            log.info("Reconnected to ElevenLabs for %s", session_id)

            mode = session.meta.get("mode", "lecture")
            mode_config = TranscriptionConfig.get_mode_config(mode)
            if mode_config.commit_strategy == "manual":
                if session.manual_commit_task is not None:
                    session.manual_commit_task.cancel()
                    try:
                        await session.manual_commit_task
                    except asyncio.CancelledError:
                        pass

                commit_interval = mode_config.commit_interval
                assert commit_interval is not None
                session.manual_commit_task = asyncio.create_task(
                    self._manual_commit_loop(session_id, commit_interval)
                )
                log.debug("Restarted manual commit loop after reconnection")

            return True
        except Exception as exc:
            error_msg = f"Failed to reconnect to ElevenLabs: {str(exc)}"
            log.error(error_msg)
            try:
                await self._send_session_event(session, "error", {"message": error_msg})
            except Exception:
                pass
            return False
            
    async def handle_binary_audio(self, session_id: str, data: bytes) -> None:
        """Handle a binary message (audio data) from the frontend client."""
        session = self.sessions.get(session_id)
        
        if LogConfig.LOG_AUDIO_CHUNKS:
            log.debug("Audio chunk: %d bytes", len(data))
        
        if session is None or not session.is_active:
            if LogConfig.VERBOSE:
                log.debug("Audio for invalid/inactive session %s", session_id)
            return

        if not await self._ensure_elevenlabs_connection(session, session_id):
            return

        try:
            await session.eleven_client.send_audio_chunk(data)
        except Exception as exc:
            error_msg = f"Failed to send audio to ElevenLabs: {str(exc)}"
            log.error(error_msg)
            
            try:
                await self._send_session_event(session, "error", {"message": error_msg})
            except Exception:
                pass

    async def handle_text_message(self, session_id: str, text: str) -> None:
        """Handle a text message coming from the frontend client."""
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            log.debug("Received text for invalid/inactive session %s", session_id)
            return

        stripped = text.strip()

        if stripped.upper().startswith("PROJECT_ID:"):
            await self._handle_project_command(session, session_id, stripped)
            return

        if stripped.upper().startswith("MODE:"):
            await self._handle_mode_command(session, session_id, stripped)
            return

        if stripped.upper() == "STOP":
            await self._handle_stop_command(session, session_id)
            return

        if stripped.upper() == "SAVE":
            log.info("Received SAVE for session %s", session_id)
            await self._save_session_to_db(session_id)
            return

        log.debug("Text from %s: %s", session_id, text[:100])
        try:
            await self._send_session_event(session, "echo", {"text": text})
        except Exception as exc:
            log.warning("Failed to send echo to %s: %s", session_id, exc)



    async def _generate_final_summary(self, session: LiveSession) -> None:
        """
        生成最终摘要（Stop 时调用）
        
        使用完整的 full transcript 生成一个总结性摘要
        
        Args:
            session: LiveSession 对象
        """
        try:
            full_transcript = "\n".join(session.transcript_parts)
            
            if not full_transcript or len(full_transcript.strip()) < 20:
                log.debug("Transcript too short, skipping final summary")
                return
            
            log.info("Generating final summary: %d sentences, %d chars",
                     len(session.transcript_parts), len(full_transcript))
            
            summary_data = await self.persistence.create_final_summary(session)
            if summary_data is None:
                if not session.db_session_id:
                    log.warning("No database session ID, cannot save final summary")
                else:
                    log.warning("Empty final summary generated")
                return

            log.info("Final summary generated: %d chars", len(summary_data["content"]))
            await self._send_session_event(session, "summary", summary_data)
            log.debug("Sent final summary to client")
        except Exception as e:
            log.error("Error in _generate_final_summary: %s", e, exc_info=True)
            if not session.db_session_id:
                log.warning("No database session ID, cannot save final summary")
            
    async def push_transcript_to_client(
        self,
        session_id: str,
        text: str,
        is_final: bool,
    ) -> None:
        """
        Send transcription text back to the frontend client.

        Called from ElevenLabs client callbacks (on_partial / on_final).

        Message format: JSON event with type "partial" or "final".
        
        Args:
            session_id: Which session should receive the text.
            text: The transcript content.
            is_final: True for committed transcript, False for partial.
        """
        # 🔧 获取 session
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            if LogConfig.VERBOSE:
                log.debug("Cannot push transcript, session %s is invalid/inactive", session_id)
            return

        msg_type = "final" if is_final else "partial"
        if is_final and text.strip():
            stripped_text = text.strip()
            session.transcript_parts.append(stripped_text)
            session.ingestion_buffer.append(stripped_text)
            
            if LogConfig.LOG_TRANSCRIPT:
                log.debug("[final] '%s' | Buffer: %d",
                          stripped_text[:60], len(session.ingestion_buffer))
            
            # 检查是否应该生成摘要
            should_trigger = self._should_generate_summary(session)
            
            if should_trigger:
                if LogConfig.LOG_SUMMARY:
                    log.info("Trigger condition met, initiating summary generation")
                try:
                    await self._send_session_event(session, "summary_started", {"is_final": False})
                except Exception as exc:
                    log.warning("Failed to send summary_started to %s: %s", session_id, exc)
                asyncio.create_task(self._generate_and_send_summary(session))
            else:
                if LogConfig.VERBOSE:
                    elapsed = time.time() - session.last_summary_time
                    log.debug("Waiting: %.1fs, %d sentences", elapsed, len(session.ingestion_buffer))
            
            # 🔇 只在 VERBOSE 模式显示状态快照
            if LogConfig.VERBOSE:
                log.debug("State: ingestion=%d, cache=%d, generating=%s",
                          len(session.ingestion_buffer),
                          len(session.context_cache), session.is_generating)

        try:
            await self._send_session_event(session, msg_type, {"text": text})
            if LogConfig.LOG_WEBSOCKET_MESSAGES:
                log.debug("Pushed %s to %s: %s", msg_type, session_id, text[:50])
                
        except Exception as exc:
            log.error("Failed to push transcript to %s: %s", session_id, exc, exc_info=True)
            return


    async def _manual_commit_loop(self, session_id: str, interval: float) -> None:
        """
        Background task for manual commit strategy.

        For discussion/presentation mode, automatically commits transcripts
        at regular intervals (default 12s based on Experiment C).

        This loop runs until the session is closed or the task is cancelled.
        """
        log.debug("Manual commit loop started for %s (interval=%ds)", session_id, interval)
        
        try:
            while True:
                await asyncio.sleep(interval)
                
                session = self.sessions.get(session_id)
                if session is None or not session.is_active:
                    log.debug("Manual commit loop: session %s no longer active", session_id)
                    break

                if session.eleven_client is not None:
                    if not session.eleven_client.is_alive():
                        log.debug("Skipping commit — connection not alive for %s", session_id)
                        continue
                    
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
            
    def _cleanup_after_summary(self, session: LiveSession, snapshot: List[str]) -> None:
        """
        清理摘要生成后的状态
        
        Args:
            session: LiveSession 对象
            snapshot: 刚处理的句子快照
        """
        # 重置生成状态
        session.is_generating = False
        
        # 更新最后生成摘要的时间
        session.last_summary_time = time.time()
        
        log.debug("Cleanup after summary: cache=%d, generating=%s, summaries=%d",
                  len(session.context_cache), session.is_generating,
                  session.summary_count)
