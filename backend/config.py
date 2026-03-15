"""Runtime configuration for the backend."""

from dataclasses import dataclass
from typing import Optional
import os
import sys
from pathlib import Path
import json
import logging

log = logging.getLogger("transcriber.config")

# 检测是否为 PyInstaller 打包
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
    APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber"
    log.info("Running as packaged app, base_dir=%s", BASE_DIR)
else:
    BASE_DIR = Path(__file__).parent
    APP_DATA_DIR = BASE_DIR
    log.info("Running in development mode, base_dir=%s", BASE_DIR)

# 创建必要的目录
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
(APP_DATA_DIR / "runs").mkdir(exist_ok=True)
(APP_DATA_DIR / "database").mkdir(exist_ok=True)

# 配置路径
MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = APP_DATA_DIR / "database" / "transcriptions.db"
RUNS_DIR = APP_DATA_DIR / "runs"

log.debug("Paths: models=%s, db=%s, runs=%s", MODELS_DIR, DATABASE_PATH, RUNS_DIR)

# ==================== API Keys Configuration ====================

def get_config_file_path():
    """获取配置文件路径"""
    home = Path.home()
    config_file = home / "Library" / "Application Support" / "RealtimeTranscriber" / "api_keys.json"
    log.debug("Config file: %s (exists: %s)", config_file, config_file.exists())
    return config_file


def load_api_keys():
    """从环境变量或配置文件加载 API Keys 和语言设置。"""
    openai_key_env = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_key_env = os.getenv("ELEVENLABS_API_KEY", "")
    hf_key_env = os.getenv("HUGGINGFACE_API_KEY", "")
    openai_key_file = ""
    elevenlabs_key_file = ""
    hf_key_file = ""
    language_file = ""
    config_file = get_config_file_path()

    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.loads(f.read())
            openai_key_file = config.get("openai_api_key", "")
            elevenlabs_key_file = config.get("elevenlabs_api_key", "")
            hf_key_file = config.get("huggingface_api_key", "")
            language_file = config.get("transcription_language", "")
        except Exception as exc:
            log.warning("Failed to load API key config: %s", exc)

    openai_key = openai_key_env or openai_key_file
    elevenlabs_key = elevenlabs_key_env or elevenlabs_key_file
    hf_key = hf_key_env or hf_key_file
    language = os.getenv("TRANSCRIPTION_LANGUAGE", language_file)
    return openai_key, elevenlabs_key, hf_key, language


# 加载配置
OPENAI_API_KEY, ELEVENLABS_API_KEY, HUGGINGFACE_API_KEY, TRANSCRIPTION_LANGUAGE = load_api_keys()
log.info("Transcription language: %s", TRANSCRIPTION_LANGUAGE or "auto-detect")

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
    """Mode-specific ElevenLabs configuration."""

    # None = auto-detect; set from TRANSCRIPTION_LANGUAGE if provided
    _lang: Optional[str] = TRANSCRIPTION_LANGUAGE or None

    LECTURE = ModeConfig(
        commit_strategy="manual",
        commit_interval=35.0,
        vad_silence_threshold_secs=None,
        vad_threshold=None,
        min_speech_duration_ms=None,
        min_silence_duration_ms=None,
    )

    DISCUSSION = ModeConfig(
        commit_strategy="vad",
        commit_interval=None,
        vad_silence_threshold_secs=0.5,
        vad_threshold=0.4,
        min_speech_duration_ms=300,
        min_silence_duration_ms=300,
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
        

# ==================== 摘要配置 ====================

class SummaryConfig:
    """摘要生成配置"""

    SUMMARY_INTERVAL_SECONDS: int = int(os.getenv("SUMMARY_INTERVAL_SECONDS", "45"))
    LOOSE_MODE_THRESHOLD: int = int(
        os.getenv("LOOSE_MODE_THRESHOLD", str(SUMMARY_INTERVAL_SECONDS + 10))
    )
    MIN_SENTENCES: int = 3
    SENTENCE_ENDERS: set = {
        "。",
        "？",
        "！",
        ".",
        "?",
        "!",
        "…",
        "......",
    }
    CONTEXT_CACHE_SIZE: int = 3

    # "" = auto / follow transcript; "en" = English; "zh" = Chinese
    LANGUAGE: str = TRANSCRIPTION_LANGUAGE

    @staticmethod
    def get_api_key() -> str:
        return OPENAI_API_KEY

    MODEL: str = "gpt-4-turbo"
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1024
    API_URL: str = "https://api.openai.com/v1/chat/completions"
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        log.info("Summary Config: interval=%ds, loose=%ds, min_sentences=%d, model=%s",
                 cls.SUMMARY_INTERVAL_SECONDS, cls.LOOSE_MODE_THRESHOLD,
                 cls.MIN_SENTENCES, cls.MODEL)


class EmbeddingConfig:
    """Embedding 配置（默认使用 Hugging Face Inference API）。"""

    PROVIDER: str = "huggingface"
    MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DIMENSION: int = 384
    BATCH_SIZE: int = 16
    TIMEOUT_SECONDS: float = 30.0
    API_BASE_URL: str = "https://api-inference.huggingface.co/pipeline/feature-extraction"

    @staticmethod
    def get_api_key() -> str:
        return HUGGINGFACE_API_KEY

    @classmethod
    def get_api_url(cls, model_name: str) -> str:
        model = model_name or cls.MODEL
        return f"{cls.API_BASE_URL}/{model}"
    
# ==================== 日志配置 ====================

class LogConfig:
    """日志输出控制配置"""

    VERBOSE: bool = True
    LOG_AUDIO_CHUNKS: bool = False
    LOG_RUN_EVENTS: bool = False
    LOG_WEBSOCKET_MESSAGES: bool = False
    LOG_TRANSCRIPT: bool = True
    LOG_SUMMARY: bool = True
    LOG_SESSION: bool = True
    LOG_DATABASE: bool = True
    
    @classmethod
    def enable_verbose(cls):
        """启用详细日志（调试模式）"""
        cls.VERBOSE = True
        cls.LOG_AUDIO_CHUNKS = True
        cls.LOG_RUN_EVENTS = True
        cls.LOG_WEBSOCKET_MESSAGES = True
        log.info("Verbose logging ENABLED")
    
    @classmethod
    def enable_quiet(cls):
        """启用安静模式（只记录重要事件）"""
        cls.VERBOSE = False
        cls.LOG_AUDIO_CHUNKS = False
        cls.LOG_RUN_EVENTS = False
        cls.LOG_WEBSOCKET_MESSAGES = False
        log.info("Quiet mode ENABLED")
    
    @classmethod
    def print_config(cls):
        """打印日志配置"""
        log.info("Log Config: verbose=%s, audio=%s, ws=%s, transcript=%s, summary=%s",
                 cls.VERBOSE, cls.LOG_AUDIO_CHUNKS, cls.LOG_WEBSOCKET_MESSAGES,
                 cls.LOG_TRANSCRIPT, cls.LOG_SUMMARY)
