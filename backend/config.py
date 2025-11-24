"""
Configuration for transcription modes.
All mode-specific settings are centralized here.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModeConfig:
    """Configuration for a specific transcription mode."""
    
    # ElevenLabs API settings
    commit_strategy: str
    commit_interval: Optional[float]  # Only used for manual strategy
    
    # VAD settings (only used for VAD strategy)
    vad_silence_threshold_secs: Optional[float]
    vad_threshold: Optional[float]
    min_speech_duration_ms: Optional[int]
    min_silence_duration_ms: Optional[int]
    
    # Audio settings (common)
    audio_format: str = "pcm_16000"
    sample_rate: int = 16000
    language_code: Optional[str] = None
    timestamps_granularity: str = "word"
    model_id: str = "scribe_v2_realtime"


class TranscriptionConfig:
    """
    Centralized configuration for all transcription modes.
    
    Modes:
    - lecture: Manual commit strategy with periodic commits (8-15s)
              Best for: Lectures, presentations, long-form speech
    
    - discussion: VAD commit strategy with pause detection
                 Best for: Conversations, interviews, Q&A sessions
    """
    
    # 🔧 Lecture Mode: Manual commit every 35 seconds
    LECTURE = ModeConfig(
        commit_strategy="manual",
        commit_interval= 35.0,  # Commit every 35 seconds
        
        # VAD not used in manual mode
        vad_silence_threshold_secs=None,
        vad_threshold=None,
        min_speech_duration_ms=None,
        min_silence_duration_ms=None,
    )
    
    # 🔧 Discussion Mode: VAD with 1.5s pause detection
    DISCUSSION = ModeConfig(
        commit_strategy="vad",
        commit_interval=None,  # Not used in VAD mode
        
        # VAD settings
        vad_silence_threshold_secs=1.5,  # Commit after 1.5s silence
        vad_threshold=0.4,               # Voice activity threshold
        min_speech_duration_ms=1000,     # Minimum speech duration (1s)
        min_silence_duration_ms=1000,    # Minimum silence duration (1s)
    )
    
    @classmethod
    def get_mode_config(cls, mode: str) -> ModeConfig:
        """
        Get configuration for the specified mode.
        
        Args:
            mode: "lecture" or "discussion"
        
        Returns:
            ModeConfig for the specified mode
        
        Raises:
            ValueError: If mode is not recognized
        """
        mode = mode.lower()
        
        if mode == "lecture":
            return cls.LECTURE
        elif mode == "discussion":
            return cls.DISCUSSION
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'lecture' or 'discussion'.")