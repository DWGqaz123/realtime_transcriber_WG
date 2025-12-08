# backend/session_manager.py (Improved Version)
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from fastapi import WebSocket
from uuid import uuid4
from config import TranscriptionConfig 
from run_logger import RunLogger
from elevenlabs_client import ElevenLabsRealtimeClient, ElevenLabsConfig
import os
import time

@dataclass
class Session:
    """
    Represents the state of a WebSocket transcription session.
    """
    id: str
    websocket: WebSocket
    eleven_client: Optional[ElevenLabsRealtimeClient] = None
    is_active: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)
    manual_commit_task: Optional[asyncio.Task] = None
    last_audio_time: float = 0.0
    
        # session ID in the database
    db_session_id: Optional[int] = None
    is_saved: bool = False
    
    # statistics
    start_time: float = field(default_factory=lambda: time.time())
    transcript_parts: List[str] = field(default_factory=list)
    


class SessionManager:
    """
    Manages all transcription sessions.

    Responsibilities:
    - Create and close Session objects.
    - Route incoming messages (text / binary) to the correct handlers.
    - Push transcription results back to the frontend.
    - Cooperate with RunLogger to record session information.
    """

    def __init__(self, run_logger: Optional[RunLogger] = None, api_key: Optional[str] = None):
        """
        Initialize the SessionManager.

        Args:
        - run_logger: Optional RunLogger to record session runs.
        - api_key: ElevenLabs API key. If None, read from environment.
        """
        self.sessions: Dict[str, Session] = {}
        self.run_logger = run_logger
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")

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

    async def create_session(self, websocket: WebSocket) -> Session:
        """
        Create a new session for the given WebSocket connection.

        - Creates the Session object with basic metadata.
        - ElevenLabs client will be created lazily when MODE is received.
        - This allows users to connect without immediately consuming resources.
        """
        session_id = str(uuid4())
        session = Session(id=session_id, websocket=websocket)
        self.sessions[session_id] = session

        if self.run_logger is not None:
            self.run_logger.start_run(session_id, meta={"status": "created"})

        print(f"[SessionManager] Created session {session_id}")
        return session

    async def close_session(self, session_id: str) -> None:
        """
        Close the session with the given ID.

        Steps:
        1. Mark session as inactive
        2. Cancel manual commit task if running
        3. Close ElevenLabs client connection
        4. Close WebSocket (if not already closed by client)
        5. Notify RunLogger
        6. Remove from sessions dictionary
        
        Note: Does NOT auto-save session data - user must explicitly call SAVE
        """
        session = self.sessions.get(session_id)
        if not session:
            print(f"[SessionManager] Session {session_id} not found, already closed?")
            return

        print(f"[SessionManager] Closing session {session_id}")
    
        
        # Mark inactive
        session.is_active = False

        # Cancel manual commit task if running
        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            try:
                await session.manual_commit_task
            except asyncio.CancelledError:
                pass
            print(f"[SessionManager] Cancelled manual commit task for {session_id}")

        # Close ElevenLabs client
        if session.eleven_client is not None:
            try:
                await session.eleven_client.close()
                print(f"[SessionManager] Closed ElevenLabs client for {session_id}")
            except Exception as exc:
                print(f"[SessionManager] Error closing ElevenLabs client: {exc}")

        # Close WebSocket if still open
        try:
            # Check if WebSocket is still connected
            if hasattr(session.websocket, 'client_state'):
                from fastapi.websockets import WebSocketState
                if session.websocket.client_state != WebSocketState.DISCONNECTED:
                    await session.websocket.close(code=1000, reason="Session closed normally")
            else:
                await session.websocket.close()
            print(f"[SessionManager] Closed WebSocket for {session_id}")
        except Exception as exc:
            print(f"[SessionManager] WebSocket already closed or error: {exc}")

        # Finish run logging
        if self.run_logger is not None:
            self.run_logger.finish_run(session_id)

        # Remove session from dictionary
        del self.sessions[session_id]
        print(f"[SessionManager] Session {session_id} fully cleaned up")
        
    
    async def _save_session_to_db(self, session_id: str) -> None:
        """
        save the current session's transcript to the database.
        """
        session = self.sessions.get(session_id)
        if session is None:
            print(f"[SessionManager] Cannot save: session {session_id} not found")
            return
        
        # check if already saved
        if session.is_saved:
            print(f"[SessionManager] Session {session_id} already saved")
            try:
                await session.websocket.send_text("[save] Session already saved")
            except:
                pass
            return
        
        # check if db_session_id exists
        if session.db_session_id is None:
            print(f"[SessionManager] Cannot save: no db_session_id for session {session_id}")
            try:
                await session.websocket.send_text("[save] ERROR: No database session")
            except:
                pass
            return
        
        # check if there is any transcript to save
        if not session.transcript_parts:
            print(f"[SessionManager] Cannot save: no transcript for session {session_id}")
            try:
                await session.websocket.send_text("[save] ERROR: No transcript to save")
            except:
                pass
            return
        
        # save to database
        try:
            from database.db import DatabaseManager
            from datetime import datetime
            
            # calculate duration
            duration = int(time.time() - session.start_time)
            
            # combine transcript parts
            full_transcript = "\n".join(session.transcript_parts)
            
            # satistics
            sentence_count = len(session.transcript_parts)
            char_count = len(full_transcript)
            
            # renew session record in database
            DatabaseManager.update_session(
                session_id=session.db_session_id,
                duration_seconds=duration,
                transcript_text=full_transcript,
                sentence_count=sentence_count,
                char_count=char_count,
                ended_at=datetime.utcnow()
            )
            
            # mark as saved
            session.is_saved = True

            print(f"[SessionManager] ✅ Saved session to DB: {duration}s, {sentence_count} sentences, {char_count} chars")

            # save to local file (if project_id available)
            project_id = session.meta.get("project_id")
            if project_id is not None:
                try:
                    # fetch project info
                    project = DatabaseManager.get_project_by_id(project_id)
                    if project:
                        # determine mode
                        mode = session.meta.get("mode", "unknown")
                        
                        # save to file
                        filepath = self._save_transcript_to_file(
                            project_name=project.name,
                            mode=mode,
                            transcript_text=full_transcript,
                            started_at=DatabaseManager.get_session_by_id(session.db_session_id).started_at.isoformat() if session.db_session_id else datetime.utcnow().isoformat()
                        )
                        
                        if filepath:
                            print(f"[SessionManager] 📄 File saved: {filepath}")
                    else:
                        print(f"[SessionManager] ⚠️ Project {project_id} not found for file export")
                except Exception as exc:
                    print(f"[SessionManager] ❌ Failed to export to file: {exc}")

            # notify client
            try:
                await session.websocket.send_text(f"[save] Session saved successfully ({sentence_count} sentences, {char_count} chars)")
            except Exception as exc:
                print(f"[SessionManager] Failed to send save confirmation: {exc}")
            
        except Exception as exc:
            print(f"[SessionManager] ❌ Failed to save session to DB: {exc}")
            import traceback
            traceback.print_exc()
            
            try:
                await session.websocket.send_text(f"[save] ERROR: {str(exc)}")
            except:
                pass
    
    def _save_transcript_to_file(
        self,
        project_name: str,
        mode: str,
        transcript_text: str,
        started_at: str) -> Optional[str]:
        """
        Save the transcript to /transcripts.
        
        Args:
            project_name: project name for directory
            mode:  (lecture/discussion)
            transcript_text: full transcript content
            started_at: start time in ISO format string
        
        Returns:
            The file path if saved successfully, else None.
        """
        try:
            from pathlib import Path
            from datetime import datetime
            
            # Base directory            
            base_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "transcripts"


            safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
            project_dir = base_dir / safe_project_name
            
            # create directories if not exist
            project_dir.mkdir(parents=True, exist_ok=True)

            
            # filename with timestamp
            # eg. 2025-12-01_14-30-15_lecture.txt
            try:
                dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
            
            timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_{mode}.txt"
            filepath = project_dir / filename
            
            # write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                # write metadata header
                f.write(f"Project: {project_name}\n")
                f.write(f"Mode: {mode}\n")
                f.write(f"Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 60 + "\n\n")
                
                # write transcript
                f.write(transcript_text)
            
            print(f"[SessionManager] 📄 Transcript saved to file: {filepath}")
            return str(filepath)
            
        except Exception as exc:
            print(f"[SessionManager] ❌ Failed to save transcript to file: {exc}")
            import traceback
            traceback.print_exc()
            return None

    async def handle_text_message(self, session_id: str, text: str) -> None:
        """Handle a text message coming from the frontend client."""
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received text for invalid/inactive session {session_id}")
            return

        stripped = text.strip()
        
        # PROJECT configuration
        if stripped.upper().startswith("PROJECT:"):
            try:
                project_id_str = stripped.split(":", 1)[1].strip()
                project_id = int(project_id_str)
                session.meta["project_id"] = project_id
                print(f"[SessionManager] Set project_id={project_id} for session {session_id}")
                
                try:
                    await session.websocket.send_text(f"[config] project set to {project_id}")
                except Exception as exc:
                    print(f"[SessionManager] Failed to send project acknowledgment: {exc}")
                
                return
            except (ValueError, IndexError) as exc:
                print(f"[SessionManager] Invalid PROJECT message: {text}, error: {exc}")
                return

        # 1) MODE configuration
        if stripped.upper().startswith("MODE:"):
            mode_raw = stripped.split(":", 1)[1].strip().lower()
            mode = "lecture" if mode_raw not in ["discussion"] else "discussion"
            session.meta["mode"] = mode

            # Skip if client already exists
            if session.eleven_client is not None:
                print(f"[SessionManager] Mode already configured for {session_id}, skipping")
                try:
                    await session.websocket.send_text(f"[config] mode already set to {mode}")
                except Exception as exc:
                    print(f"[SessionManager] Failed to send acknowledgment: {exc}")
                return

            # Build config and connect
            print(f"[SessionManager] Setting mode to '{mode}' for session {session_id}")
            
            if self.run_logger is not None:
                self.run_logger.log_event(session_id, {"type": "mode_set", "mode": mode})

            try:
                config = self._build_elevenlabs_config_for_mode(mode)
                eleven_client = ElevenLabsRealtimeClient(self._api_key, config)

                # Setup callbacks
                def on_partial(text: str):
                    asyncio.create_task(
                        self.push_transcript_to_client(session_id, text, is_final=False)
                    )

                def on_final(text: str):
                    asyncio.create_task(
                        self.push_transcript_to_client(session_id, text, is_final=True)
                    )

                eleven_client.on_partial = on_partial
                eleven_client.on_final = on_final

                # Connect
                await eleven_client.connect()
                session.eleven_client = eleven_client
                success_msg = f"[config] mode set to {mode}, connected to ElevenLabs"
                print(f"[SessionManager] {success_msg}")
                
                # 🔧 新增：创建数据库会话记录

                try:
                    from database.db import DatabaseManager
                    project_id = session.meta.get("project_id")
                    
                    if project_id is not None:
                        # 🔧 新增：检查是否已有未保存的会话
                        if session.db_session_id is None or session.is_saved:
                            # 创建新的数据库会话
                            db_session = DatabaseManager.create_session(
                                project_id=project_id,
                                mode=mode
                            )
                            session.db_session_id = db_session.id
                            session.is_saved = False
                            session.start_time = time.time()  # 重置开始时间
                            session.transcript_parts = []      # 清空转录（如果是新会话）
                            print(f"[SessionManager] Created new DB session: {db_session.id} for project {project_id}")
                        else:
                            # 复用现有会话
                            print(f"[SessionManager] Reusing existing DB session: {session.db_session_id}")
                            # 不清空 transcript_parts，继续累积
                    else:
                        print(f"[SessionManager] ⚠️ No project_id in meta, skipping DB session creation")
                except Exception as exc:
                    print(f"[SessionManager] Failed to create/reuse DB session: {exc}")

                # 🔧 使用集中配置的 commit_interval
                if config.commit_strategy == "manual":
                    mode_config = TranscriptionConfig.get_mode_config(mode)
                    commit_interval = mode_config.commit_interval
                    assert commit_interval is not None, \
                    f"Manual mode requires commit_interval, but got None for mode '{mode}'"
                    
                    session.manual_commit_task = asyncio.create_task(
                        self._manual_commit_loop(session_id, commit_interval)
                    )
                    print(f"[SessionManager] Started manual commit loop with {commit_interval}s interval")

            except Exception as exc:
                error_msg = f"[error] Failed to connect to ElevenLabs: {str(exc)}"
                print(f"[SessionManager] {error_msg}")
                session.meta["connection_error"] = str(exc)
                
                try:
                    await session.websocket.send_text(error_msg)
                except Exception:
                    pass
                return

            return

        # 2) STOP signal
        if stripped.upper() == "STOP":
            print(f"[SessionManager] Received STOP signal from {session_id}")
            
            if session.manual_commit_task is not None:
                session.manual_commit_task.cancel()
                session.manual_commit_task = None
                print(f"[SessionManager] Cancelled manual commit task for {session_id} on STOP ")
            
            if session.eleven_client is not None:
                await session.eleven_client.close()
                session.eleven_client = None
                print(f"[SessionManager] Closed ElevenLabs client for {session_id} on STOP ")
                
            try:
                await session.websocket.send_text("[config] recording stopped, session kept alive")
            except Exception as exc:
                print(f"[SessionManager] Failed to send stop acknowledgment: {exc}")
    
            return
        
        
        
        if stripped.upper() == "SAVE":
            print(f"[SessionManager] Received SAVE for session {session_id}")
            await self._save_session_to_db(session_id)
            return
    
        # 3) Other text: simple echo
        print(f"[SessionManager] Text from {session_id}: {text}")
        echo_text = f"Echo: {text}"

        try:
            await session.websocket.send_text(echo_text)
        except Exception as exc:
            print(f"[SessionManager] Failed to send echo to {session_id}: {exc}")

        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {"type": "text_echo", "received": text, "sent": echo_text}
            )

    async def handle_binary_audio(self, session_id: str, data: bytes) -> None:
        """Handle a binary message (audio data) from the frontend client."""
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received audio for invalid/inactive session {session_id}")
            return

        if session.eleven_client is None:
            print(f"[SessionManager] Audio received but ElevenLabs client not initialized for {session_id}")
            print("[SessionManager] Client should send MODE: first")
            return

        # Record last audio time
        session.last_audio_time = time.time()

        # Check and reconnect if needed
        if not session.eleven_client.is_alive():
            print(f"[SessionManager] ⚠️ ElevenLabs connection is dead for {session_id}, attempting to reconnect...")
            
            idle_time = time.time() - session.last_audio_time
            print(f"[SessionManager] Connection was idle for {idle_time:.1f}s")
            
            try:
                await session.eleven_client.close()
                await session.eleven_client.connect()
                print(f"[SessionManager] ✅ Reconnected to ElevenLabs for {session_id}")
                
                # 🔧 重连后重启 manual commit 任务，使用集中配置
                mode = session.meta.get("mode", "lecture")
                mode_config = TranscriptionConfig.get_mode_config(mode)
                
                if mode_config.commit_strategy == "manual":
                    # Cancel old task
                    if session.manual_commit_task is not None:
                        session.manual_commit_task.cancel()
                        try:
                            await session.manual_commit_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Start new task with centralized config
                    commit_interval = mode_config.commit_interval
                    assert commit_interval is not None, \
                    f"Manual mode requires commit_interval, but got None for mode '{mode}'"
                    session.manual_commit_task = asyncio.create_task(
                        self._manual_commit_loop(session_id, commit_interval)
                    )
                    
                    print(f"[SessionManager] Restarted manual commit loop after reconnection")
                
            except Exception as e:
                error_msg = f"Failed to reconnect to ElevenLabs: {str(e)}"
                print(f"[SessionManager] ❌ {error_msg}")
                
                try:
                    await session.websocket.send_text(f"[error] {error_msg}")
                except Exception:
                    pass
                return

        print(f"[SessionManager] Audio chunk from {session_id}: {len(data)} bytes")

        # Forward to ElevenLabs
        try:
            await session.eleven_client.send_audio_chunk(data)
        except Exception as exc:
            error_msg = f"Failed to send audio to ElevenLabs: {str(exc)}"
            print(f"[SessionManager] {error_msg}")
            
            try:
                await session.websocket.send_text(f"[error] {error_msg}")
            except Exception:
                pass

        # Log for analysis
        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {"type": "audio_chunk", "size": len(data)}
            )


    async def push_transcript_to_client(
        self,
        session_id: str,
        text: str,
        is_final: bool,
    ) -> None:
        """
        Send transcription text back to the frontend client.

        Called from ElevenLabs client callbacks (on_partial / on_final).

        Message format: "[partial] text" or "[final] text"
        - Frontend can parse this to update current subtitle vs full transcript
        
        Args:
        - session_id: Which session should receive the text.
        - text: The transcript content.
        - is_final: True for committed transcript, False for partial.
        """
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Cannot push transcript, session {session_id} is invalid/inactive")
            return

        msg_type = "final" if is_final else "partial"
        payload = f"[{msg_type}] {text}"
        
        # record final transcripts
        if is_final and text.strip():
            session.transcript_parts.append(text.strip())
            print(f"[SessionManager] 📝 Recorded final transcript: {len(text)} chars")

        try:
            await session.websocket.send_text(message)
            print(f"[SessionManager] Pushed {prefix} transcript to {session_id}: {text[:50]}...")
        except Exception as exc:
            print(f"[SessionManager] Failed to push transcript: {exc}")

        # Log event
        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {"type": "transcript", "is_final": is_final, "text": text}
            )

        try:
            await session.websocket.send_text(payload)
            print(f"[SessionManager] Pushed {msg_type} transcript to {session_id}: {text[:50]}...")
        except Exception as exc:
            print(f"[SessionManager] Failed to push transcript to {session_id}: {exc}")

        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {"type": "transcript", "is_final": is_final, "text": text}
            )

    async def _manual_commit_loop(self, session_id: str, interval: float) -> None:
        """
        Background task for manual commit strategy.

        For discussion/presentation mode, automatically commits transcripts
        at regular intervals (default 12s based on Experiment C).

        This loop runs until the session is closed or the task is cancelled.
        """
        print(f"[SessionManager] Manual commit loop started for {session_id} (interval={interval}s)")
        
        try:
            while True:
                await asyncio.sleep(interval)
                
                session = self.sessions.get(session_id)
                if session is None or not session.is_active:
                    print(f"[SessionManager] Manual commit loop: session {session_id} no longer active")
                    break

                if session.eleven_client is not None:
                    # 🔧 新增：检查连接
                    if not session.eleven_client.is_alive():
                        print(f"[SessionManager] ⚠️ Skipping commit - connection not alive for {session_id}")
                        continue
                    
                    try:
                        # 🔧 修改：实际发送 commit
                        await session.eleven_client.send_commit()
                        print(f"[SessionManager] ✅ Manual commit sent for {session_id}")
                        
                        if self.run_logger is not None:
                            self.run_logger.log_event(
                                session_id,
                                {"type": "manual_commit", "interval": interval}
                            )
                        
                    except Exception as exc:
                        print(f"[SessionManager] ❌ Error sending manual commit: {exc}")

        except asyncio.CancelledError:
            print(f"[SessionManager] Manual commit loop cancelled for {session_id}")
            raise
        except Exception as exc:
            print(f"[SessionManager] Unexpected error in manual commit loop: {exc}")