"""
Configuration for transcription modes.
All mode-specific settings are centralized here.

==================== 摘要配置说明 ====================

📋 配置修改指南：

1. 时间触发配置
   - SUMMARY_INTERVAL_SECONDS: 标准触发间隔
     * 默认 300 秒（5 分钟）
     * 建议范围：180-600 秒（3-10 分钟）
   
   - LOOSE_MODE_THRESHOLD: 宽松模式阈值
     * 默认 330 秒（5 分 30 秒）
     * 应该比 SUMMARY_INTERVAL_SECONDS 大 30-60 秒
     * 超过此时间后，将忽略语义完整性检查

2. 内容触发配置
   - MIN_SENTENCES: 最少句子数
     * 默认 5 句
     * 建议范围：3-10 句
   
   - MIN_CHARS: 最少字符数（暂未使用）
     * 默认 200 字符
     * 预留功能，暂未在触发逻辑中使用

3. 上下文管理配置
   - CONTEXT_CACHE_SIZE: 上下文缓存大小
     * 默认 3 句
     * 保留最后 N 句作为下次摘要的上下文
     * 建议范围：2-5 句

4. OpenAI API 配置
   - MODEL: 使用的模型
     * 默认 gpt-4-turbo
     * 可选：gpt-4, gpt-3.5-turbo
   
   - TEMPERATURE: 温度参数
     * 默认 0.3（较低，输出更确定）
     * 范围：0.0-1.0
   
   - MAX_TOKENS: 最大输出长度
     * 默认 1024
     * 足够生成 3-5 个知识点
"""

from dataclasses import dataclass
from typing import Optional
import os

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
        vad_silence_threshold_secs= 0.5,  # Commit after 1 s silence
        vad_threshold=0.4,               # Voice activity threshold
        min_speech_duration_ms=300,     # Minimum speech duration 
        min_silence_duration_ms=300,    # Minimum silence duration 
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
    
    
    # ==================== 时间触发配置 ====================
        
    # 标准触发时间间隔（秒）
    # 默认：5分钟 = 300秒
    SUMMARY_INTERVAL_SECONDS: int = 30
    
    # 宽松模式触发时间（秒）
    # 超过此时间后降低触发要求（忽略语义完整性检查）
    # 默认：5分30秒 = 330秒
    LOOSE_MODE_THRESHOLD: int = 40
    
    # ==================== 内容触发配置 ====================
    
    # 最少句子数
    # 触发摘要需要累积至少这么多句子 5
    MIN_SENTENCES: int = 5
    
    # 最少字符数（可选，暂未使用）
    MIN_CHARS: int = 200
    
    # ==================== 语义完整性配置 ====================
    
    # 句子结束符集合
    SENTENCE_ENDERS: set = {
        '。',  # 中文句号
        '？',  # 中文问号
        '！',  # 中文感叹号
        '.',   # 英文句号
        '?',   # 英文问号
        '!',   # 英文感叹号
        '…',   # 省略号
        '......',  # 连续省略号
    }
    
    # ==================== 上下文管理配置 ====================
    
    # 上下文缓存大小（保留最后 N 句作为下次的上下文）3
    CONTEXT_CACHE_SIZE: int = 3
    
    # ==================== OpenAI API 配置 ====================
    
    # API 密钥
    # OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # 同时使用类方法延迟读取API_KEY
    @classmethod
    def get_api_key(cls) -> str:
        """
        获取 OpenAI API 密钥（延迟读取，确保 .env 已加载）
        
        Returns:
            str: API 密钥，如果未设置则返回空字符串
        """
        return os.getenv("OPENAI_API_KEY", "")
    
    # 模型选择
    MODEL: str = "gpt-4-turbo"
    
    # 温度参数（0-1，越低越确定性）
    TEMPERATURE: float = 0.3
    
    # 最大 token 数
    MAX_TOKENS: int = 1024
    
    API_URL: str = "https://api.openai.com/v1/chat/completions"
    
    @classmethod
    def validate(cls):
        """验证配置"""
        errors = []
        
        # 验证时间配置
        if cls.SUMMARY_INTERVAL_SECONDS <= 0:
            errors.append("SUMMARY_INTERVAL_SECONDS must be positive")
        
        if cls.LOOSE_MODE_THRESHOLD < cls.SUMMARY_INTERVAL_SECONDS:
            errors.append("LOOSE_MODE_THRESHOLD should be >= SUMMARY_INTERVAL_SECONDS")
        
        # 验证内容配置
        if cls.MIN_SENTENCES <= 0:
            errors.append("MIN_SENTENCES must be positive")
        
        if cls.MIN_CHARS < 0:
            errors.append("MIN_CHARS cannot be negative")
        
        # 验证上下文配置
        if cls.CONTEXT_CACHE_SIZE <= 0:
            errors.append("CONTEXT_CACHE_SIZE must be positive")
        
        # 验证 API 配置
        api_key = cls.get_api_key()
        if not api_key:
            errors.append("OPENAI_API_KEY is not set in environment variables")
        
        if errors:
            error_msg = "Summary configuration errors:\n  " + "\n  ".join(errors)
            raise ValueError(error_msg)
        
        return True
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "="*60)
        print("📋 Summary Configuration")
        print("="*60)
        
        print(f"⏱️  Trigger Intervals:")
        print(f"   Standard interval: {cls.SUMMARY_INTERVAL_SECONDS}s ({cls.SUMMARY_INTERVAL_SECONDS/60:.1f} min)")
        print(f"   Loose mode threshold: {cls.LOOSE_MODE_THRESHOLD}s ({cls.LOOSE_MODE_THRESHOLD/60:.1f} min)")
        
        print(f"\n📝 Content Requirements:")
        print(f"   Min sentences: {cls.MIN_SENTENCES}")
        print(f"   Min characters: {cls.MIN_CHARS}")
        
        print(f"\n🔤 Sentence Enders:")
        enders_str = ', '.join(f"'{e}'" for e in sorted(cls.SENTENCE_ENDERS))
        print(f"   {enders_str}")
        
        print(f"\n💾 Context:")
        print(f"   Cache size: {cls.CONTEXT_CACHE_SIZE} sentences")
        
        print(f"\n🤖 OpenAI API:")
        print(f"   Model: {cls.MODEL}")
        print(f"   Temperature: {cls.TEMPERATURE}")
        print(f"   Max tokens: {cls.MAX_TOKENS}")
        print(f"   API URL: {cls.API_URL}")
        
        # 🔧 使用方法读取 API 密钥
        api_key = cls.get_api_key()
        if api_key:
            # 只显示前 7 个字符
            masked_key = api_key[:7] + "..." if len(api_key) > 7 else "***"
            print(f"   API key: ✅ Set ({masked_key})")
        else:
            print(f"   API key: ❌ Missing")
        
        print("="*60 + "\n")
    
# ==================== 日志配置 ====================

class LogConfig:
    """日志输出控制配置"""
    
    # ==================== 日志级别 ====================
    
    # 是否启用详细日志（开发模式）
    VERBOSE: bool = True
    
    # 是否记录音频块日志
    LOG_AUDIO_CHUNKS: bool = False
    
    # 是否记录 RunLogger 事件
    LOG_RUN_EVENTS: bool = False
    
    # 是否记录 WebSocket 消息
    LOG_WEBSOCKET_MESSAGES: bool = False
    
    # 是否记录转录文本
    LOG_TRANSCRIPT: bool = True
    
    # 是否记录摘要生成
    LOG_SUMMARY: bool = True
    
    # 是否记录会话管理
    LOG_SESSION: bool = True
    
    # 是否记录数据库操作
    LOG_DATABASE: bool = True
    
    @classmethod
    def enable_verbose(cls):
        """启用详细日志（调试模式）"""
        cls.VERBOSE = True
        cls.LOG_AUDIO_CHUNKS = True
        cls.LOG_RUN_EVENTS = True
        cls.LOG_WEBSOCKET_MESSAGES = True
        print("🔊 Verbose logging ENABLED")
    
    @classmethod
    def enable_quiet(cls):
        """启用安静模式（只记录重要事件）"""
        cls.VERBOSE = False
        cls.LOG_AUDIO_CHUNKS = False
        cls.LOG_RUN_EVENTS = False
        cls.LOG_WEBSOCKET_MESSAGES = False
        print("🔇 Quiet mode ENABLED (audio chunks/events hidden)")
    
    @classmethod
    def print_config(cls):
        """打印日志配置"""
        print("\n" + "="*60)
        print("📋 Logging Configuration")
        print("="*60)
        print(f"Verbose mode: {cls.VERBOSE}")
        print(f"Audio chunks: {cls.LOG_AUDIO_CHUNKS}")
        print(f"Run events: {cls.LOG_RUN_EVENTS}")
        print(f"WebSocket messages: {cls.LOG_WEBSOCKET_MESSAGES}")
        print(f"Transcript: {cls.LOG_TRANSCRIPT}")
        print(f"Summary: {cls.LOG_SUMMARY}")
        print(f"Session: {cls.LOG_SESSION}")
        print(f"Database: {cls.LOG_DATABASE}")
        print("="*60 + "\n")