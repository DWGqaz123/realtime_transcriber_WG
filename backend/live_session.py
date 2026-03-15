import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from fastapi import WebSocket
from elevenlabs_client import ElevenLabsRealtimeClient

log = logging.getLogger("transcriber.session")


@dataclass
class LiveSession:
    """Single recording session state."""

    id: str
    websocket: WebSocket
    meta: dict
    started_at: datetime

    start_time: float = field(default_factory=time.time)

    # Transcript
    transcript_parts: List[str] = field(default_factory=list)

    # Summary buffers
    ingestion_buffer: List[str] = field(default_factory=list)
    context_cache: List[str] = field(default_factory=list)

    # Summary state
    is_generating: bool = False
    last_summary_time: float = field(default_factory=time.time)
    summary_count: int = 0
    next_start_sentence_idx: int = 0

    # Session state
    is_active: bool = True
    is_saved: bool = False
    db_session_id: Optional[int] = None

    # ElevenLabs
    eleven_client: Optional[ElevenLabsRealtimeClient] = None
    manual_commit_task: Optional[asyncio.Task] = None

    def __post_init__(self):
        log.debug("Created session: %s", self.id)
