# backend/elevenlabs_client.py (Improved Version)

import base64
import json
import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, Any
from urllib.parse import urlencode
import httpx
import websockets

@dataclass
class ElevenLabsConfig:
    """
    Configuration for an ElevenLabs realtime transcription session.

    - audio_format: e.g. "pcm_16000"
    - sample_rate: must match the actual audio content (8000-48000)
    - language_code: ISO-639 code or None for auto-detect
    - timestamps_granularity: "word" if you want word-level timestamps
    - mode: app-level mode, "lecture" or "discussion"
    - model_id: Scribe realtime model ID
    - commit_strategy: "manual" or "vad"
    - vad_*: VAD-related settings used when commit_strategy = "vad"
    """
    audio_format: str = "pcm_16000"
    sample_rate: int = 16000
    language_code: Optional[str] = None
    timestamps_granularity: str = "word"
    mode: str = "lecture"

    model_id: str = "scribe_v2_realtime"
    commit_strategy: str = "vad"
    vad_silence_threshold_secs: Optional[float] = 1.5
    vad_threshold: Optional[float] = 0.4
    min_speech_duration_ms: Optional[int] = 250


class ElevenLabsRealtimeClient:
    """
    Encapsulates communication with ElevenLabs Scribe v2 Realtime.

    - Connects to wss://api.elevenlabs.io/v1/speech-to-text/realtime
    - Sends audio chunks as base64
    - Receives partial / committed transcripts
    - Supports both VAD and manual commit strategies
    """

    def __init__(self, api_key: str, config: ElevenLabsConfig):
        self.api_key = api_key
        self.config = config

        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None

        self._ws: Any = None
        self._connected: bool = False
        self._last_chunk_had_audio: bool = False  # Track if we've sent audio since last commit

    async def _fetch_single_use_token(self) -> str:
        """
        Request a single-use token from ElevenLabs for realtime scribe.

        Endpoint:
        POST https://api.elevenlabs.io/v1/single-use-token/realtime_scribe
        Header: xi-api-key: ELEVENLABS_API_KEY
        
        Returns:
        - Single-use authentication token for WebSocket connection
        """
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

        url = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"
        headers = {"xi-api-key": self.api_key}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        token = data.get("token")
        if not token:
            raise RuntimeError("Failed to obtain single-use token from ElevenLabs")

        return token

    async def connect(self) -> None:
        """
        Establish a realtime connection to ElevenLabs.

        Auth strategy:
        1) Use API key to request a single-use token via HTTPS.
        2) Pass that token as `token` in the WebSocket query params.
        3) Connect to WebSocket with full configuration.
        
        Raises:
        - RuntimeError if no API key is provided
        - Various exceptions if connection fails
        """
        if self._connected:
            print("[ElevenLabsRealtimeClient] Already connected, skipping")
            return

        if not self.api_key:
            raise RuntimeError("No API key provided, cannot connect")

        # 1) Get single-use token
        try:
            token = await self._fetch_single_use_token()
            print("[ElevenLabsRealtimeClient] Obtained single-use token")
        except Exception as exc:
            print(f"[ElevenLabsRealtimeClient] Failed to fetch token: {exc}")
            raise

        # 2) Build WebSocket URL with all parameters
        base_url = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

        params: dict[str, str] = {
            "model_id": self.config.model_id,
            "audio_format": self.config.audio_format,
            "commit_strategy": self.config.commit_strategy,
            "token": token,
        }

        if self.config.language_code:
            params["language_code"] = self.config.language_code

        if self.config.timestamps_granularity == "word":
            params["include_timestamps"] = "true"

        # Only add VAD parameters if using VAD strategy
        if self.config.commit_strategy == "vad":
            if self.config.vad_silence_threshold_secs is not None:
                params["vad_silence_threshold_secs"] = str(self.config.vad_silence_threshold_secs)
            if self.config.vad_threshold is not None:
                params["vad_threshold"] = str(self.config.vad_threshold)
            if self.config.min_speech_duration_ms is not None:
                params["min_speech_duration_ms"] = str(self.config.min_speech_duration_ms)

        url = f"{base_url}?{urlencode(params)}"
        print(f"[ElevenLabsRealtimeClient] Connecting to {url[:100]}...")

        # 3) Connect to WebSocket
        try:
            self._ws = await websockets.connect(
                url,
                max_size=16 * 1024 * 1024,  # 16MB max message size
            )
        except Exception as exc:
            print(f"[ElevenLabsRealtimeClient] Failed to connect: {exc}")
            self._connected = False
            self._ws = None
            raise

        self._connected = True
        print("[ElevenLabsRealtimeClient] Connected successfully")

        # Start background receive loop
        asyncio.create_task(self._receive_loop())

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """
        Send a chunk of audio to ElevenLabs.

        - Audio must be mono, PCM, sample rate given in config.
        - Encoded as base64 and wrapped in a JSON message.
        
        For manual commit strategy:
        - Audio chunks are sent with commit=False
        - Actual commits are triggered separately via send_commit()
        
        Args:
        - audio_bytes: Raw PCM audio data (16-bit, mono, 16kHz recommended)
        """
        if not self._connected or self._ws is None:
            print("[ElevenLabsRealtimeClient] Cannot send audio, not connected")
            return

        b64_audio = base64.b64encode(audio_bytes).decode("ascii")

        # Build payload according to ElevenLabs API spec
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": b64_audio,
            "sample_rate": self.config.sample_rate,
            # For manual strategy: always False, commit separately
            # For VAD strategy: False, let server decide when to commit
            "commit": False,
        }

        try:
            await self._ws.send(json.dumps(payload))
            self._last_chunk_had_audio = True
        except Exception as exc:
            print(f"[ElevenLabsRealtimeClient] Failed to send audio chunk: {exc}")
            raise

    async def send_commit(self) -> None:
        """
        Send an explicit commit signal for manual commit strategy.

        This tells ElevenLabs to finalize the current transcript segment
        and treat it as a committed result.

        Should be called periodically when using manual commit strategy.
        
        No-op if not connected or if commit strategy is VAD.
        """
        if not self._connected or self._ws is None:
            print("[ElevenLabsRealtimeClient] Cannot send commit, not connected")
            return

        if self.config.commit_strategy != "manual":
            print("[ElevenLabsRealtimeClient] Commit signal only relevant for manual strategy")
            return

        if not self._last_chunk_had_audio:
            # No audio sent since last commit, skip
            return

        # Send empty audio chunk with commit=True
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": "",  # Empty audio
            "sample_rate": self.config.sample_rate,
            "commit": True,
        }

        try:
            await self._ws.send(json.dumps(payload))
            self._last_chunk_had_audio = False
            print("[ElevenLabsRealtimeClient] Sent manual commit signal")
        except Exception as exc:
            print(f"[ElevenLabsRealtimeClient] Failed to send commit: {exc}")
            raise

    async def _receive_loop(self) -> None:
        """
        Background loop to receive messages from ElevenLabs.

        Handles:
        - session_started / sessionStarted
        - partial_transcript / partialTranscript
        - committed_transcript / committedTranscript / committedTranscriptWithTimestamps
        - error messages
        
        Calls registered callbacks (on_partial / on_final) when transcripts arrive.
        """
        if self._ws is None:
            return

        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except Exception as exc:
                    print(f"[ElevenLabsRealtimeClient] Failed to parse message: {exc}")
                    continue

                msg_type = msg.get("message_type") or msg.get("type")

                if msg_type in ("session_started", "sessionStarted"):
                    session_id = msg.get("session_id", "unknown")
                    print(f"[ElevenLabsRealtimeClient] Session started: {session_id}")

                elif msg_type in ("partial_transcript", "partialTranscript"):
                    text = msg.get("transcript") or msg.get("text") or ""
                    if text and self.on_partial:
                        self.on_partial(text)

                elif msg_type in (
                    "committed_transcript",
                    "committedTranscript",
                    "committedTranscriptWithTimestamps",
                ):
                    text = msg.get("transcript") or msg.get("text") or ""
                    if text and self.on_final:
                        self.on_final(text)

                elif msg_type and "error" in msg_type.lower():
                    error_msg = msg.get("message", msg.get("error", str(msg)))
                    print(f"[ElevenLabsRealtimeClient] Error from server: {error_msg}")

                else:
                    # Unknown message type, log for debugging
                    print(f"[ElevenLabsRealtimeClient] Unknown message type: {msg_type}")

        except asyncio.CancelledError:
            print("[ElevenLabsRealtimeClient] Receive loop cancelled")
            raise
        except Exception as exc:
            print(f"[ElevenLabsRealtimeClient] Receive loop error: {exc}")
            self._connected = False

    async def close(self) -> None:
        """
        Close the connection to ElevenLabs and clean up resources.
        
        - Closes WebSocket connection gracefully
        - Marks client as disconnected
        - Safe to call multiple times
        """
        if self._ws is not None:
            try:
                await self._ws.close()
                print("[ElevenLabsRealtimeClient] WebSocket closed")
            except Exception as exc:
                print(f"[ElevenLabsRealtimeClient] Error while closing: {exc}")
            finally:
                self._ws = None
                
        self._connected = False
        print("[ElevenLabsRealtimeClient] Client disconnected")