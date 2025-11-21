import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from fastapi import WebSocket
from uuid import uuid4

from run_logger import RunLogger
from elevenlabs_client import ElevenLabsRealtimeClient, ElevenLabsConfig
import os


@dataclass
class Session:
    """
    Represents the state of a WebSocket transcription session.

    Fields:
    - id: Unique session ID (e.g., a UUID string).
    - websocket: The WebSocket connection to the frontend client.
    - eleven_client: Client that talks to ElevenLabs Realtime.
    - is_active: Whether this session is still active.
    - meta: Arbitrary metadata (mode, language, start time, errors, etc.).
    - manual_commit_task: Background task for manual commit strategy.
    """
    id: str
    websocket: WebSocket
    eleven_client: Optional[ElevenLabsRealtimeClient] = None
    is_active: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)
    manual_commit_task: Optional[asyncio.Task] = None


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

        Modes:
        - lecture: VAD commit strategy, optimal for long monologues
        - discussion: Manual commit strategy with 12s interval (optimal for presentations)
        
        Based on Experiment C results:
        - 8s: Dictation mode (short responses)
        - 12s: Presentation mode (best for structured speech)
        - 20s: Lecture mode (continuous professor lectures)
        """
        mode = mode.lower()
        
        if mode == "discussion":
            # Discussion mode uses manual commit with optimal 12s interval
            return ElevenLabsConfig(
                audio_format="pcm_16000",
                sample_rate=16000,
                language_code=None,
                timestamps_granularity="word",
                mode=mode,
                commit_strategy="manual",
                # Manual strategy doesn't use VAD parameters
                vad_silence_threshold_secs=None,
                vad_threshold=None,
                min_speech_duration_ms=None,
            )
        else:
            # Lecture mode uses VAD for natural pause detection
            return ElevenLabsConfig(
                audio_format="pcm_16000",
                sample_rate=16000,
                language_code=None,
                timestamps_granularity="word",
                mode="lecture",
                commit_strategy="vad",
                vad_silence_threshold_secs=1.5,  # Treat 1.5s silence as end of segment
                vad_threshold=0.4,               # Voice activity detection threshold
                min_speech_duration_ms=250,      # Minimum speech duration
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
        """
        session = self.sessions.get(session_id)
        if not session:
            print(f"[SessionManager] Session {session_id} not found, already closed?")
            return

        session.is_active = False
        print(f"[SessionManager] Closing session {session_id}")

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

    async def handle_text_message(self, session_id: str, text: str) -> None:
        """
        Handle a text message coming from the frontend client.

        Supported message formats:
        1. "MODE:lecture" or "MODE:discussion"
           - Configures the transcription mode and connects to ElevenLabs
        2. "STOP"
           - Signals end of recording (currently just logged)
        3. Other text
           - Simple echo for testing (can be removed in production)
        """
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received text for invalid/inactive session {session_id}")
            return

        stripped = text.strip()

        # 1) MODE configuration
        if stripped.upper().startswith("MODE:"):
            mode_raw = stripped.split(":", 1)[1].strip().lower()
            mode = "lecture" if mode_raw != "discussion" else "discussion"
            session.meta["mode"] = mode

            print(f"[SessionManager] Setting mode to '{mode}' for session {session_id}")

            if self.run_logger is not None:
                self.run_logger.log_event(
                    session_id,
                    {"type": "mode_set", "mode": mode}
                )

            # Create and connect ElevenLabs client if not already present
            if session.eleven_client is None:
                if not self._api_key:
                    error_msg = "[error] ELEVENLABS_API_KEY not configured"
                    print(f"[SessionManager] {error_msg}")
                    try:
                        await session.websocket.send_text(error_msg)
                    except Exception:
                        pass
                    return

                # Build configuration based on mode
                config = self._build_elevenlabs_config_for_mode(mode)
                eleven_client = ElevenLabsRealtimeClient(api_key=self._api_key, config=config)

                # Bind callbacks to push transcripts to frontend
                def handle_partial(text_: str) -> None:
                    asyncio.create_task(
                        self.push_transcript_to_client(session_id, text_, is_final=False)
                    )

                def handle_final(text_: str) -> None:
                    asyncio.create_task(
                        self.push_transcript_to_client(session_id, text_, is_final=True)
                    )

                eleven_client.on_partial = handle_partial
                eleven_client.on_final = handle_final

                # Attempt to connect to ElevenLabs
                try:
                    await eleven_client.connect()
                    session.eleven_client = eleven_client
                    success_msg = f"[config] mode set to {mode}, connected to ElevenLabs"
                    print(f"[SessionManager] {success_msg}")
                    
                    try:
                        await session.websocket.send_text(success_msg)
                    except Exception as exc:
                        print(f"[SessionManager] Failed to send success message: {exc}")

                    # Start manual commit task if needed
                    if config.commit_strategy == "manual":
                        commit_interval = 12.0  # Optimal for presentations (from Experiment C)
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

            else:
                # Client already exists, just acknowledge
                try:
                    await session.websocket.send_text(f"[config] mode already set to {mode}")
                except Exception as exc:
                    print(f"[SessionManager] Failed to send acknowledgment: {exc}")

            return

        # 2) STOP signal
        if stripped.upper() == "STOP":
            print(f"[SessionManager] Received STOP signal from {session_id}")
            if self.run_logger is not None:
                self.run_logger.log_event(session_id, {"type": "stop_signal"})
            
            # Optionally send a final flush to ElevenLabs
            # (Currently not implemented in elevenlabs_client)
            
            try:
                await session.websocket.send_text("[info] Recording stopped")
            except Exception as exc:
                print(f"[SessionManager] Failed to send stop acknowledgment: {exc}")
            return

        # 3) Other text: simple echo for testing
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
        """
        Handle a binary message (audio data) from the frontend client.

        Flow:
        1. Validate session exists and is active
        2. Forward audio chunk to ElevenLabs client
        3. Log the event for analysis
        
        Note: Audio should be 16kHz, mono, PCM format (converted in Swift app)
        """
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received audio for invalid/inactive session {session_id}")
            return

        # Check if ElevenLabs client is connected
        if session.eleven_client is None:
            print(f"[SessionManager] Audio received but ElevenLabs client not initialized for {session_id}")
            print("[SessionManager] Client should send MODE: first")
            return

        print(f"[SessionManager] Audio chunk from {session_id}: {len(data)} bytes")

        # Forward to ElevenLabs
        try:
            await session.eleven_client.send_audio_chunk(data)
        except Exception as exc:
            error_msg = f"Failed to send audio to ElevenLabs: {str(exc)}"
            print(f"[SessionManager] {error_msg}")
            
            # Notify client of error
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
                    try:
                        # Send a commit signal to ElevenLabs
                        # Note: This requires implementing send_commit() in elevenlabs_client.py
                        # For now, we log it
                        print(f"[SessionManager] Manual commit triggered for {session_id}")
                        
                        if self.run_logger is not None:
                            self.run_logger.log_event(
                                session_id,
                                {"type": "manual_commit", "interval": interval}
                            )
                        
                        # TODO: Implement actual commit sending
                        # await session.eleven_client.send_commit()
                        
                    except Exception as exc:
                        print(f"[SessionManager] Error in manual commit: {exc}")

        except asyncio.CancelledError:
            print(f"[SessionManager] Manual commit loop cancelled for {session_id}")
            raise
        except Exception as exc:
            print(f"[SessionManager] Unexpected error in manual commit loop: {exc}")