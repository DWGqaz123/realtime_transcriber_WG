"""Runtime configuration for the backend."""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import sys
from pathlib import Path
import json
import logging

log = logging.getLogger("transcriber.config")

# 检测是否为 PyInstaller 打包。BASE_DIR 目前只用于启动日志——数据路径由
# 各模块自行解析（DatabaseManager、FAISSIndexManager、SessionPersistence
# 都直接指向 ~/Library/Application Support/RealtimeTranscriber/）。
#
# 这里曾经还有 APP_DATA_DIR / DATABASE_PATH / MODELS_DIR / RUNS_DIR 四个常量
# 和三次 mkdir，但没有任何模块引用它们：models 目录在改用远程 embedding 后
# 就不存在了，RunLogger 从未被实例化，DATABASE_PATH 指向的 transcriptions.db
# 也从来没被打开过（真正的库是 transcripts.db）。开发模式下 APP_DATA_DIR
# 等于 backend/，那次 mkdir 会去"创建" backend/database/ 这个源码目录，
# 正是它让人误以为那里是数据目录，进而写出了会静默忽略源码的 gitignore 规则。
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
    log.info("Running as packaged app, base_dir=%s", BASE_DIR)
else:
    BASE_DIR = Path(__file__).parent
    log.info("Running in development mode, base_dir=%s", BASE_DIR)

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
    secondary_file: list = []
    config_file = get_config_file_path()

    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.loads(f.read())
            openai_key_file = config.get("openai_api_key", "")
            elevenlabs_key_file = config.get("elevenlabs_api_key", "")
            hf_key_file = config.get("huggingface_api_key", "")
            language_file = config.get("transcription_language", "")
            secondary_file = config.get("secondary_languages", []) or []
        except Exception as exc:
            log.warning("Failed to load API key config: %s", exc)

    openai_key = openai_key_env or openai_key_file
    elevenlabs_key = elevenlabs_key_env or elevenlabs_key_file
    hf_key = hf_key_env or hf_key_file
    language = os.getenv("TRANSCRIPTION_LANGUAGE", language_file)

    # 环境变量里用逗号分隔（"zh,ja"），配置文件里是数组。
    # 注意 ElevenLabs 只接受重复 query 参数，不接受逗号分隔的单个值，
    # 所以这里统一解析成 list，拼 URL 时靠 urlencode(doseq=True) 展开。
    secondary_env = os.getenv("SECONDARY_LANGUAGES", "")
    if secondary_env:
        secondary = [x.strip() for x in secondary_env.split(",") if x.strip()]
    else:
        secondary = [str(x).strip() for x in secondary_file if str(x).strip()]

    return openai_key, elevenlabs_key, hf_key, language, secondary


# 加载配置
(
    OPENAI_API_KEY,
    ELEVENLABS_API_KEY,
    HUGGINGFACE_API_KEY,
    TRANSCRIPTION_LANGUAGE,
    SECONDARY_LANGUAGES,
) = load_api_keys()

# scribe_v2_realtime_turbo 自 2026-07 起可用，延迟更低；
# 设 STT_MODEL_ID=scribe_v2_realtime 可回退。
STT_MODEL_ID: str = os.getenv("STT_MODEL_ID", "scribe_v2_realtime_turbo")

log.info(
    "Transcription language: %s (secondary: %s), STT model: %s",
    TRANSCRIPTION_LANGUAGE or "auto-detect",
    ", ".join(SECONDARY_LANGUAGES) if SECONDARY_LANGUAGES else "none",
    STT_MODEL_ID,
)

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
    # 额外允许出现的语言（中英混说场景）。ElevenLabs 要求重复 query 参数。
    secondary_languages: List[str] = field(default_factory=list)
    timestamps_granularity: str = "word"
    model_id: str = STT_MODEL_ID


class TranscriptionConfig:
    """Mode-specific ElevenLabs configuration."""

    # None = auto-detect; set from TRANSCRIPTION_LANGUAGE if provided
    _lang: Optional[str] = TRANSCRIPTION_LANGUAGE or None

    LECTURE = ModeConfig(
        commit_strategy="manual",
        commit_interval=35.0,
        language_code=_lang,
        secondary_languages=list(SECONDARY_LANGUAGES),
        vad_silence_threshold_secs=None,
        vad_threshold=None,
        min_speech_duration_ms=None,
        min_silence_duration_ms=None,
    )

    DISCUSSION = ModeConfig(
        commit_strategy="vad",
        commit_interval=None,
        language_code=_lang,
        secondary_languages=list(SECONDARY_LANGUAGES),
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

    MODEL: str = os.getenv("SUMMARY_MODEL", "gpt-5.6-luna")

    # 仅对 gpt-4 / gpt-3.5 这代模型生效。gpt-5.x / gpt-6 / o 系列只接受
    # temperature 的默认值，传 0.3 会被拒（见 SummaryService._build_payload）。
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
    """Embedding 配置。

    默认使用 OpenAI embeddings：摘要功能已经要求用户配置 OPENAI_API_KEY，
    复用它可以避免再申请第二个服务的 key。设 EMBEDDING_PROVIDER=huggingface
    可切回 HF（此时需要单独提供 HUGGINGFACE_API_KEY）。
    """

    PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    _IS_OPENAI: bool = PROVIDER != "huggingface"

    MODEL: str = (
        "text-embedding-3-small"
        if _IS_OPENAI
        else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    DIMENSION: int = 1536 if _IS_OPENAI else 384
    BATCH_SIZE: int = 16
    TIMEOUT_SECONDS: float = 30.0

    _OPENAI_API_URL: str = "https://api.openai.com/v1/embeddings"
    # api-inference.huggingface.co 已下线，新入口是 router.huggingface.co
    _HF_BASE_URL: str = "https://router.huggingface.co/hf-inference/models"

    @classmethod
    def is_openai(cls) -> bool:
        return cls._IS_OPENAI

    @classmethod
    def get_api_key(cls) -> str:
        return OPENAI_API_KEY if cls._IS_OPENAI else HUGGINGFACE_API_KEY

    @classmethod
    def get_api_url(cls, model_name: str = "") -> str:
        if cls._IS_OPENAI:
            return cls._OPENAI_API_URL
        model = model_name or cls.MODEL
        return f"{cls._HF_BASE_URL}/{model}/pipeline/feature-extraction"

    @classmethod
    def print_config(cls):
        log.info(
            "Embedding Config: provider=%s, model=%s, dim=%d, key=%s",
            cls.PROVIDER, cls.MODEL, cls.DIMENSION,
            "ok" if cls.get_api_key() else "MISSING",
        )
