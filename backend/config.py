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
import sys
from pathlib import Path
import json

# 检测是否为 PyInstaller 打包
if getattr(sys, 'frozen', False):
    # 打包后的路径
    BASE_DIR = Path(sys._MEIPASS)
    # 数据文件需要放在用户目录
    APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber"
    print(f"🔍 Running as packaged app")
    print(f"📦 Base dir: {BASE_DIR}")
else:
    # 开发环境
    BASE_DIR = Path(__file__).parent
    APP_DATA_DIR = BASE_DIR
    print(f"🔧 Running in development mode")
    print(f"📁 Base dir: {BASE_DIR}")

# 创建必要的目录
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
(APP_DATA_DIR / "runs").mkdir(exist_ok=True)
(APP_DATA_DIR / "database").mkdir(exist_ok=True)

# 配置路径
MODELS_DIR = BASE_DIR / "models"
DATABASE_PATH = APP_DATA_DIR / "database" / "transcriptions.db"
RUNS_DIR = APP_DATA_DIR / "runs"

print(f"📁 Models dir: {MODELS_DIR}")
print(f"💾 Database: {DATABASE_PATH}")
print(f"📝 Runs dir: {RUNS_DIR}")

# ==================== API Keys Configuration ====================

def get_config_file_path():
    """获取配置文件路径"""
    home = Path.home()
    config_file = home / "Library" / "Application Support" / "RealtimeTranscriber" / "api_keys.json"
    
    print("\n" + "="*60)
    print("📂 STEP 2: Looking for Config File")
    print("="*60)
    print(f"Config file path: {config_file}")
    print(f"File exists: {config_file.exists()}")
    
    if config_file.exists():
        print(f"File size: {config_file.stat().st_size} bytes")
        print(f"File permissions: {oct(config_file.stat().st_mode)}")
    
    print("="*60 + "\n")
    
    return config_file


def load_api_keys():
    """从多个来源加载 API Keys"""
    
    print("\n" + "="*60)
    print("🔑 STEP 3: Loading API Keys")
    print("="*60)
    
    # 1. 检查环境变量
    openai_key_env = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_key_env = os.getenv("ELEVENLABS_API_KEY", "")
    
    print("\n1️⃣ Environment Variables:")
    print(f"   OPENAI_API_KEY: {'SET (' + str(len(openai_key_env)) + ' chars)' if openai_key_env else 'NOT SET'}")
    if openai_key_env:
        print(f"   Preview: {openai_key_env[:10]}...{openai_key_env[-4:]}")
    
    print(f"   ELEVENLABS_API_KEY: {'SET (' + str(len(elevenlabs_key_env)) + ' chars)' if elevenlabs_key_env else 'NOT SET'}")
    if elevenlabs_key_env:
        print(f"   Preview: {elevenlabs_key_env[:10]}...{elevenlabs_key_env[-4:]}")
    
    # 2. 从配置文件读取
    openai_key_file = ""
    elevenlabs_key_file = ""
    
    config_file = get_config_file_path()
    
    print("\n2️⃣ Config File:")
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_content = f.read()
                print(f"   File content ({len(file_content)} chars):")
                print(f"   {file_content[:200]}...")
                
                # 重新解析
                config = json.loads(file_content)
                
                openai_key_file = config.get("openai_api_key", "")
                elevenlabs_key_file = config.get("elevenlabs_api_key", "")
                
                print(f"\n   Parsed keys:")
                print(f"   openai_api_key: {'SET (' + str(len(openai_key_file)) + ' chars)' if openai_key_file else 'NOT SET'}")
                if openai_key_file:
                    print(f"   Preview: {openai_key_file[:10]}...{openai_key_file[-4:]}")
                
                print(f"   elevenlabs_api_key: {'SET (' + str(len(elevenlabs_key_file)) + ' chars)' if elevenlabs_key_file else 'NOT SET'}")
                if elevenlabs_key_file:
                    print(f"   Preview: {elevenlabs_key_file[:10]}...{elevenlabs_key_file[-4:]}")
                
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ❌ Config file not found")
    
    # 3. 选择最终值
    openai_key = openai_key_env or openai_key_file
    elevenlabs_key = elevenlabs_key_env or elevenlabs_key_file
    
    print("\n3️⃣ Final Values:")
    print(f"   OPENAI_API_KEY: {'SET (' + str(len(openai_key)) + ' chars)' if openai_key else 'NOT SET'}")
    if openai_key:
        print(f"   Source: {'Environment' if openai_key_env else 'Config File'}")
        print(f"   Preview: {openai_key[:10]}...{openai_key[-4:]}")
    
    print(f"   ELEVENLABS_API_KEY: {'SET (' + str(len(elevenlabs_key)) + ' chars)' if elevenlabs_key else 'NOT SET'}")
    if elevenlabs_key:
        print(f"   Source: {'Environment' if elevenlabs_key_env else 'Config File'}")
        print(f"   Preview: {elevenlabs_key[:10]}...{elevenlabs_key[-4:]}")
    
    print("="*60 + "\n")
    
    return openai_key, elevenlabs_key


# 加载 API Keys
OPENAI_API_KEY, ELEVENLABS_API_KEY = load_api_keys()


def validate_api_keys():
    """验证 API Keys"""
    missing_keys = []
    
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if not ELEVENLABS_API_KEY:
        missing_keys.append("ELEVENLABS_API_KEY")
    
    if missing_keys:
        print("\n" + "="*60)
        print("⚠️  WARNING: Missing API Keys")
        print("="*60)
        for key in missing_keys:
            print(f"   ❌ {key} is not set")
        print(f"\nPlease configure API Keys in the app Settings.")
        print(f"Config file: {get_config_file_path()}")
        print("="*60 + "\n")
        return False
    else:
        print("\n" + "="*60)
        print("✅ API Keys Configured")
        print("="*60)
        print(f"   ✅ OPENAI_API_KEY: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")
        print(f"   ✅ ELEVENLABS_API_KEY: {ELEVENLABS_API_KEY[:10]}...{ELEVENLABS_API_KEY[-4:]}")
        print("="*60 + "\n")
        return True
    

# 服务器配置
HOST = "127.0.0.1"
PORT = 8000

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
    SUMMARY_INTERVAL_SECONDS: int = 120
    
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
    # 🔧 API Key 从环境变量获取
    @staticmethod
    def get_api_key() -> str:
        """获取 OpenAI API Key"""
        return OPENAI_API_KEY
    
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