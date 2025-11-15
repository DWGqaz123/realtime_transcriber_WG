# backend/elevenlabs_client.py

class ElevenLabsRealtimeClient:
    """
    Stub implementation for now.

    Real implementation will:
    - Manage a WebSocket connection to ElevenLabs Realtime API.
    - Send audio chunks.
    - Receive partial/final transcripts.
    """

    def __init__(self, api_key: str, model: str = "scribe_v2", language: str | None = None):
        self.api_key = api_key
        self.model = model
        self.language = language

    async def connect(self) -> None:
        pass

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        pass

    async def close(self) -> None:
        pass