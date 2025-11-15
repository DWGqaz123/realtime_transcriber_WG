# backend/session_manager.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from fastapi import WebSocket
from uuid import uuid4

from elevenlabs_client import ElevenLabsRealtimeClient
from run_logger import RunLogger


@dataclass
class Session:
    """
    Represents the state of a WebSocket transcription session.

    Fields:
    - id: Unique session ID (e.g., a UUID string).
    - websocket: The WebSocket connection to the frontend client.
    - eleven_client: Client that talks to ElevenLabs Realtime (not used in phase 1).
    - is_active: Whether this session is still active.
    - meta: Arbitrary metadata for future use (language, start time, etc.).
    """
    id: str
    websocket: WebSocket
    eleven_client: Optional[ElevenLabsRealtimeClient] = None
    is_active: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    Manages all transcription sessions.

    Responsibilities:
    - Create and close Session objects.
    - Route incoming messages (text / binary) to the correct handlers.
    - Push transcription results back to the frontend.
    - Cooperate with RunLogger to record session information.
    """

    def __init__(self, run_logger: Optional[RunLogger] = None):
        """
        Initialize the SessionManager.

        Args:
        - run_logger: Optional RunLogger to record session runs.
        """
        self.sessions: Dict[str, Session] = {}
        self.run_logger = run_logger

    async def create_session(self, websocket: WebSocket) -> Session:
        """
        Create a new session for the given WebSocket connection.

        Purpose:
        - Generate a new session ID.
        - Store a Session instance in the internal dictionary.
        - Optionally record the start of a run in RunLogger.

        Returns:
        - The created Session object.
        """
        session_id = str(uuid4())
        session = Session(id=session_id, websocket=websocket)
        self.sessions[session_id] = session

        if self.run_logger is not None:
            self.run_logger.start_run(session_id, meta={"status": "started"})

        print(f"[SessionManager] Created session {session_id}")
        return session

    async def close_session(self, session_id: str) -> None:
        """
        Close the session with the given ID.

        Purpose:
        - Mark the Session as inactive.
        - Close any external resources (e.g., ElevenLabs client).
        - Close the WebSocket if needed.
        - Notify RunLogger that the run has finished.
        """
        session = self.sessions.get(session_id)
        if not session:
            return

        session.is_active = False

        # Close ElevenLabs client if present (not used in phase 1).
        if session.eleven_client is not None:
            try:
                await session.eleven_client.close()
            except Exception as exc:
                print(f"[SessionManager] Error closing ElevenLabs client: {exc}")

        # Finish run logging.
        if self.run_logger is not None:
            self.run_logger.finish_run(session_id)

        # Optionally close the WebSocket (frontend usually initiates the close).
        try:
            await session.websocket.close()
        except Exception:
            # It may already be closed by the client; ignore errors.
            pass

        # Remove session from dictionary.
        del self.sessions[session_id]
        print(f"[SessionManager] Closed session {session_id}")

    async def handle_text_message(self, session_id: str, text: str) -> None:
        """
        Handle a text message coming from the frontend client.

        Current phase (v1):
        - Simply echo the received text back to the client.
        - Also prints to the server log for debugging.

        Future phases:
        - Interpret special commands (e.g. "START", "STOP", "SET_LANGUAGE").
        - Possibly forward text to ElevenLabs if the API supports that.

        Args:
        - session_id: ID of the session that sent the message.
        - text: The text message received from the client.
        """
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received text for invalid or inactive session {session_id}")
            return

        print(f"[SessionManager] Text from {session_id}: {text}")

        # Simple echo behavior: send the message back with an 'Echo: ' prefix.
        echo_text = f"Echo: {text}"

        try:
            await session.websocket.send_text(echo_text)
        except Exception as exc:
            print(f"[SessionManager] Failed to send echo to {session_id}: {exc}")

        # Optionally log the event via RunLogger.
        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {
                    "type": "text_echo",
                    "received": text,
                    "sent": echo_text,
                },
            )

    async def handle_binary_audio(self, session_id: str, data: bytes) -> None:
        """
        Handle a binary message (audio data) from the frontend client.

        Current phase:
        - Only logs the size of the data.
        - Does not forward anything to ElevenLabs yet.

        Future phases:
        - Forward audio chunks to ElevenLabsRealtimeClient.send_audio_chunk().
        """
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received audio for invalid or inactive session {session_id}")
            return

        print(f"[SessionManager] Received audio chunk from {session_id}, size={len(data)} bytes")

        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {
                    "type": "audio_chunk",
                    "size": len(data),
                },
            )

    async def push_transcript_to_client(
        self,
        session_id: str,
        text: str,
        is_final: bool,
    ) -> None:
        """
        Send transcription text back to the frontend client.

        Purpose:
        - This will be called from ElevenLabsRealtimeClient (or from here)
          when a partial or final transcript is produced.
        - The message format can be JSON, e.g. including a "type" field.

        Args:
        - session_id: Which session should receive the text.
        - text: The transcript content.
        - is_final: True for final result, False for partial.
        """
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Cannot push transcript, session {session_id} is invalid or inactive")
            return

        msg_type = "final" if is_final else "partial"
        payload = f"[{msg_type}] {text}"

        try:
            await session.websocket.send_text(payload)
        except Exception as exc:
            print(f"[SessionManager] Failed to push transcript to {session_id}: {exc}")

        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {
                    "type": "transcript",
                    "is_final": is_final,
                    "text": text,
                },
            )