# backend/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from session_manager import SessionManager
from run_logger import RunLogger
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root (one level up from backend/)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)  # This will load ELEVENLABS_API_KEY into environment

app = FastAPI()

# Create a single, shared SessionManager instance for the whole app.
# In a more advanced setup you could use dependency injection with lifespans,
# but for now a simple global instance is enough.
base_runs_dir = Path("runs")
run_logger = RunLogger(base_dir=base_runs_dir)
eleven_api_key = os.getenv("ELEVENLABS_API_KEY", "")
session_manager = SessionManager(run_logger=run_logger, api_key=eleven_api_key)



def get_session_manager() -> SessionManager:
    """
    Dependency injector for SessionManager.

    Purpose:
    - Provide a single SessionManager instance to all routes that need it.
    - Central place to customize how SessionManager is created in the future.
    """
    return session_manager


@app.get("/ping")
async def ping():
    """
    Simple health-check endpoint.

    Purpose:
    - Allow the frontend or developer to verify that the backend is running.
    - You can later extend this to return more debugging information.
    """
    return {"message": "pong"}


@app.websocket("/ws/transcribe")
async def websocket_endpoint(
    websocket: WebSocket,
    manager: SessionManager = Depends(get_session_manager),
):
    """
    WebSocket endpoint for transcription.

    Purpose:
    - Accept WebSocket connections from the macOS client.
    - Create a new Session for each connection via SessionManager.
    - Receive messages in a loop:
        * Text messages: handled by SessionManager.handle_text_message().
        * Binary messages: later used for audio, handled by handle_binary_audio().
    - On disconnect, close the session and perform cleanup.

    Current phase:
    - Only supports text messages and echoes them back to the client.
    - This is enough to verify end-to-end connectivity (frontend <-> backend).
    """
    await websocket.accept()
    # Create a new session for this WebSocket connection.
    session = await manager.create_session(websocket)
    session_id = session.id

    try:
        while True:
            message = await websocket.receive()

            # We explicitly handle both "text" and "bytes" types here to prepare
            # for the next phase where binary audio will be sent.
            if "text" in message and message["text"] is not None:
                text = message["text"]
                await manager.handle_text_message(session_id, text)
            elif "bytes" in message and message["bytes"] is not None:
                data = message["bytes"]
                # Audio handling will be implemented later.
                await manager.handle_binary_audio(session_id, data)

    except WebSocketDisconnect:
        # Client disconnected normally.
        await manager.close_session(session_id)
    except Exception as exc:
        # Any unexpected error: log it and close the session.
        print(f"Unexpected error in WebSocket endpoint: {exc}")
        await manager.close_session(session_id)