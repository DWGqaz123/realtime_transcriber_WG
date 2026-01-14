# backend/session_manager.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from fastapi import WebSocket
from uuid import uuid4
from config import TranscriptionConfig, SummaryConfig, LogConfig
from summary_service import SummaryService
from run_logger import RunLogger
from elevenlabs_client import ElevenLabsRealtimeClient, ElevenLabsConfig
import os
import time
import json
from summary_service import get_summary_service
from database.db import DatabaseManager
from datetime import datetime
from indexing_service import get_indexing_service


@dataclass
class LiveSession:
    """会话数据类 - 管理单个录音会话的状态"""
    
    id: str
    websocket: WebSocket
    meta: dict
    started_at: datetime
    
    # statistics
    start_time: float = field(default_factory=lambda: time.time())
    transcript_parts: List[str] = field(default_factory=list)
    
    # ==================== 转录相关 ====================
    transcript_parts: List[str] = field(default_factory=list)
    
    # ==================== 双容器架构 ====================
    ingestion_buffer: List[str] = field(default_factory=list)
    processing_snapshot: List[str] = field(default_factory=list)
    context_cache: List[str] = field(default_factory=list)
    
    # 向后兼容
    summary_buffer: List[str] = field(default_factory=list)
    last_context: str = ""
    
    # ==================== 状态管理 ====================
    is_generating: bool = False
    last_summary_time: float = field(default_factory=lambda: time.time())
    sentence_count_since_last: int = 0
    summary_sentence_start_idx: int = 0
    generated_summaries: List[int] = field(default_factory=list)
    next_start_sentence_idx: int = 0
    
    # ==================== 会话状态 ====================
    is_active: bool = True
    is_saved: bool = False
    db_session_id: Optional[int] = None
    duration_seconds: int = 0
    
    # ==================== ElevenLabs 相关 ====================
    # 🔧 添加缺失的字段
    eleven_client: Optional[ElevenLabsRealtimeClient] = None  # ElevenLabs 客户端实例
    manual_commit_task: Optional[asyncio.Task] = None
    last_audio_time: float = 0.0

    
    def __post_init__(self):
        """初始化后的设置"""
        print(f"✨ Created new session: {self.id}")
        print(f"   Ingestion buffer: {len(self.ingestion_buffer)}")
        print(f"   Context cache: {len(self.context_cache)}")
        print(f"   Is generating: {self.is_generating}")


class SessionManager:
    """
    Manages all transcription sessions.

    Responsibilities:
    - Create and close Session objects.
    - Route incoming messages (text / binary) to the correct handlers.
    - Push transcription results back to the frontend.
    - Cooperate with RunLogger to record session information.
    """

    def __init__(self, api_key: Optional[str] = None, run_logger: Optional[RunLogger] = None):
        """
        Initialize the SessionManager.

        Args:
            api_key: ElevenLabs API key. If None, read from environment.
            run_logger: Optional RunLogger to record session runs.
        """
        
        print("\n" + "="*60)
        print("🔧 STEP 6: SessionManager __init__")
        print("="*60)
        
        self.sessions: Dict[str, LiveSession] = {}
        self.run_logger = run_logger
        
        # 🔧 从多个来源尝试获取 API key
        env_key = os.getenv("ELEVENLABS_API_KEY", "")
        
        print(f"📊 API Key Sources:")
        print(f"   1. Constructor param: {'[PROVIDED, ' + str(len(api_key)) + ' chars]' if api_key else '[EMPTY]'}")
        if api_key:
            print(f"      Preview: {api_key[:10]}...{api_key[-4:]}")
        
        print(f"   2. Environment var:   {'[SET, ' + str(len(env_key)) + ' chars]' if env_key else '[NOT SET]'}")
        if env_key:
            print(f"      Preview: {env_key[:10]}...{env_key[-4:]}")
        
        # 🔧 使用提供的 api_key，否则使用环境变量
        self._api_key = api_key or env_key
        
        print(f"   3. Final api_key:     {'[SET, ' + str(len(self._api_key)) + ' chars]' if self._api_key else '[NOT SET]'}")
        
        if self._api_key:
            print(f"   ✅ API Key preview: {self._api_key[:10]}...{self._api_key[-4:]}")
        else:
            print(f"   ❌ WARNING: No ElevenLabs API key available!")
        
        print(f"{'='*60}\n")
        
        # 🔧 初始化索引服务
        print("🗂️  Initializing IndexingService...")
        self.indexing_service = get_indexing_service()
        print("✅ IndexingService initialized\n")
        
        # 🔧 初始化摘要服务
        print("🤖 Initializing SummaryService...")
        try:
            from summary_service import SummaryService
            self.summary_service = SummaryService()
            print("✅ SummaryService initialized\n")
        except Exception as e:
            print(f"❌ Failed to initialize SummaryService: {e}")
            self.summary_service = None
        
        # 🔧 打印最终状态
        print("\n" + "="*60)
        print("✨ SessionManager Initialization Complete")
        print("="*60)
        print(f"   Run logger: {'enabled' if run_logger else 'disabled'}")
        print(f"   ElevenLabs API key: {'configured ✅' if self._api_key else 'missing ❌'}")
        
        # 检查 OpenAI API Key（用于摘要）
        from config import OPENAI_API_KEY
        print(f"   OpenAI API key: {'configured ✅' if OPENAI_API_KEY else 'missing ❌'}")
        
        print(f"   Summary service: {'initialized ✅' if self.summary_service else 'not configured ❌'}")
        print(f"   Indexing service: initialized ✅")
        print(f"{'='*60}\n")
    
    def _validate_phase1(self):
        """验证 Phase 1 数据结构升级"""
        print("\n🧪 Phase 1 Validation")
        print("="*60)
        
        cache_size = SummaryConfig.CONTEXT_CACHE_SIZE
        # 创建测试 session
        from unittest.mock import MagicMock
        test_ws = MagicMock()
        
        test_session = LiveSession(
            id="test-validation",
            websocket=test_ws,
            meta={"mode": "lecture"},
            started_at=datetime.now()
        )
        
        # 验证新字段存在
        checks = [
            ("ingestion_buffer", hasattr(test_session, 'ingestion_buffer')),
            ("processing_snapshot", hasattr(test_session, 'processing_snapshot')),
            ("context_cache", hasattr(test_session, 'context_cache')),
            ("is_generating", hasattr(test_session, 'is_generating')),
            ("sentence_count_since_last", hasattr(test_session, 'sentence_count_since_last')),
        ]
        
        all_passed = True
        for field_name, exists in checks:
            status = "✅" if exists else "❌"
            print(f"{status} {field_name}: {exists}")
            if not exists:
                all_passed = False
        
        # 验证辅助方法存在
        methods = [
            "_has_sentence_ender",
            "_update_context_cache",
            "_atomic_transfer",
            "_format_context_for_prompt",
            "_format_snapshot_for_prompt",
        ]
        
        print("\nHelper methods:")
        for method_name in methods:
            exists = hasattr(self, method_name)
            status = "✅" if exists else "❌"
            print(f"{status} {method_name}: {exists}")
            if not exists:
                all_passed = False
        
        # 测试原子转移
        print("\nTesting atomic transfer:")
        test_session.ingestion_buffer = ["句子1", "句子2", "句子3"]
        snapshot = self._atomic_transfer(test_session)
        
        assert len(snapshot) == 3, "Snapshot size mismatch"
        assert len(test_session.ingestion_buffer) == 0, "Buffer not cleared"
        assert test_session.is_generating == True, "Lock not set"
        
        print("✅ Atomic transfer test passed")
        
        # 测试句子结束符检查
        print("\nTesting sentence ender detection:")
        test_cases = [
            ("这是一句话。", True),
            ("这是一句话？", True),
            ("这是一句话！", True),
            ("This is a sentence.", True),
            ("This is incomplete", False),
            ("", False),
        ]
        
        for text, expected in test_cases:
            result = self._has_sentence_ender(text)
            status = "✅" if result == expected else "❌"
            print(f"{status} '{text}' → {result} (expected: {expected})")
            if result != expected:
                all_passed = False
        
        # 测试上下文缓存更新
        print("\nTesting context cache update:")
        sentences = ["句1", "句2", "句3", "句4", "句5"]
        self._update_context_cache(test_session, sentences)
        
        assert len(test_session.context_cache) == cache_size, "Cache size mismatch"
        assert test_session.context_cache == ["句3", "句4", "句5"], "Cache content mismatch"
        print("✅ Context cache test passed")
        
        print("\n" + "="*60)
        if all_passed:
            print("✅ Phase 1 Validation: ALL PASSED")
        else:
            print("❌ Phase 1 Validation: SOME CHECKS FAILED")
        print("="*60 + "\n")
    
    def _has_sentence_ender(self, text: str) -> bool:
        """
        检查文本是否以句子结束符结尾
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: 是否以结束符结尾
        """
        if not text:
            return False
        
        text = text.strip()
        
        # 检查是否以任意结束符结尾
        for ender in SummaryConfig.SENTENCE_ENDERS:
            if text.endswith(ender):
                return True
        
        return False
    
    def _update_context_cache(self, session: LiveSession, sentences: List[str]) -> None:
        """
        更新上下文缓存（保留最后 N 句）
        
        Args:
            session: 会话对象
            sentences: 新处理的句子列表
        """
        if not sentences:
            return
        
        # 取最后 CONTEXT_CACHE_SIZE 句
        cache_size = SummaryConfig.CONTEXT_CACHE_SIZE
        session.context_cache = sentences[-cache_size:]
        
        print(f"🔄 Updated context cache: {len(session.context_cache)} sentences")
        for i, sent in enumerate(session.context_cache, 1):
            preview = sent[:50] + "..." if len(sent) > 50 else sent
            print(f"   [{i}] {preview}")
    
    def _atomic_transfer(self, session: LiveSession) -> List[str]:
        """
        原子转移：从接收缓冲区复制到处理快照
        
        这是一个关键操作，需要在 <1ms 内完成：
        1. 复制 ingestion_buffer 到 processing_snapshot
        2. 清空 ingestion_buffer
        3. 设置生成锁
        4. 重置计时器
        
        Args:
            session: 会话对象
            
        Returns:
            List[str]: 处理快照的引用
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"⚡ ATOMIC TRANSFER START")
        print(f"{'='*60}")
        
        # Step 1: 复制（决策 2：使用 list.copy()）
        session.processing_snapshot = session.ingestion_buffer.copy()
        snapshot_size = len(session.processing_snapshot)
        
        # Step 2: 清空接收缓冲区
        session.ingestion_buffer.clear()
        
        # Step 3: 设置生成锁
        session.is_generating = True
        
        # Step 4: 重置计时器
        session.last_summary_time = time.time()
        
        # Step 5: 重置句子计数
        session.sentence_count_since_last = 0
        
        elapsed = (time.time() - start_time) * 1000  # 转换为毫秒
        
        print(f"✅ ATOMIC TRANSFER COMPLETE")
        print(f"   Snapshot size: {snapshot_size} sentences")
        print(f"   Buffer cleared: {len(session.ingestion_buffer)} remaining")
        print(f"   Is generating: {session.is_generating}")
        print(f"   Time elapsed: {elapsed:.3f}ms")
        print(f"{'='*60}\n")
        
        return session.processing_snapshot
    
    def _format_context_for_prompt(self, session: LiveSession) -> str:
        """
        格式化上下文缓存为 Prompt 字符串
        
        Args:
            session: 会话对象
            
        Returns:
            str: 格式化的上下文字符串
        """
        if not session.context_cache:
            return ""
        
        return "\n".join(session.context_cache)
    
    def _format_snapshot_for_prompt(self, snapshot: List[str]) -> str:
        """
        格式化处理快照为 Prompt 字符串
        
        Args:
            snapshot: 处理快照列表
            
        Returns:
            str: 格式化的快照字符串
        """
        return "\n".join(snapshot)
    
    async def _generate_and_send_summary(self, session: LiveSession) -> None:
        """
        Phase 2 版本：生成摘要的核心方法
        
        Args:
            session: LiveSession 对象
        """
        
        session_id = session.id  # 🔧 从 session 对象获取 session_id
        
        print(f"\n{'='*60}")
        print(f"🤖 SUMMARY GENERATION START")
        print(f"   Session: {session_id}")
        print(f"   DB Session: {session.db_session_id}")
        print(f"{'='*60}\n")
        
        # ==================== Phase 1: 原子转移 ====================
        print(f"\n{'='*60}")
        print(f"⚡ ATOMIC TRANSFER START")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 🔒 原子操作：转移数据
        snapshot = session.ingestion_buffer.copy()
        session.ingestion_buffer.clear()
        session.processing_snapshot = snapshot
        session.is_generating = True
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # 转换为毫秒
        
        print(f"✅ ATOMIC TRANSFER COMPLETE")
        print(f"   Snapshot size: {len(snapshot)} sentences")
        print(f"   Buffer cleared: {len(session.ingestion_buffer)} remaining")
        print(f"   Is generating: {session.is_generating}")
        print(f"   Time elapsed: {elapsed:.3f}ms")
        print(f"{'='*60}\n")
        
        if not snapshot:
            print(f"⚠️  Empty snapshot, aborting summary generation")
            session.is_generating = False
            return
        
        # ==================== Phase 2: 准备上下文和内容 ====================
        print(f"📸 Snapshot captured: {len(snapshot)} sentences")
        
        context = "\n".join(session.context_cache)
        current_text = "\n".join(snapshot)
        
        print(f"📝 Prepared content:")
        print(f"   Context: {len(context)} chars ({len(session.context_cache)} sentences)")
        print(f"   Current: {len(current_text)} chars ({len(snapshot)} sentences)")
        
        if context:
            print(f"   Context preview: {context[:100]}...")
        print(f"   Current preview: {current_text[:100]}...")
        
        # ==================== Phase 3: 调用 SummaryService ====================
        try:
            mode = session.meta.get("mode", "lecture")
            
            print(f"🤖 Calling OpenAI API...")
            print(f"   Mode: {mode}")
            print(f"   Model: gpt-4-turbo")
            
            print(f"\n{'='*60}")
            print(f"🤖 Generating summary with OpenAI API")
            print(f"   Model: gpt-4-turbo")
            print(f"   Input length: {len(current_text)} chars")
            print(f"{'='*60}\n")
            
            summary_content = await self.summary_service.generate_summary(
                buffer_text=current_text,
                context=context,
                mode=mode
            )
            
            if not summary_content:
                print(f"⚠️  Empty summary from OpenAI, aborting")
                self._cleanup_after_summary(session, snapshot)
                return
            
            print(f"✅ Summary generated successfully")
            print(f"   Output length: {len(summary_content)} chars")
            print(f"   Preview: {summary_content[:150]}...")
            
        except Exception as e:
            print(f"❌ Summary generation failed: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_after_summary(session, snapshot)
            return
        
        # ==================== Phase 4: 保存到数据库 ====================
        if session.db_session_id:
            try:
                from database.db import DatabaseManager
                
                project_id = session.meta.get("project_id")
                start_idx = session.next_start_sentence_idx
                end_idx = start_idx + len(snapshot)
                
                duration = int(time.time() - session.last_summary_time)
                source_text = current_text[:500]  # 保存前500字符作为来源
                
                summary = DatabaseManager.create_summary(
                    session_id=session.db_session_id,
                    project_id=project_id,
                    content=summary_content,
                    source_text=source_text,
                    start_sentence_idx=start_idx,
                    end_sentence_idx=end_idx,
                    duration_seconds=duration
                )
                
                session.generated_summaries.append(summary.id)
                
                print(f"✅ [DB] Created summary: id={summary.id}, session={session.db_session_id}")
                print(f"✅ [DB] Saved summary: id={summary.id}")
                print(f"   Duration: {duration}s")
                print(f"   Sentences: {start_idx} → {end_idx}")
                
                # 🔧 ==================== 发送给前端（修改这里）====================
                try:
                    summary_data = {
                        "id": summary.id,  # 🔧 添加数据库 ID
                        "content": summary_content,
                        "created_at": summary.created_at.isoformat() if summary.created_at else datetime.now().isoformat(),  # 🔧 添加创建时间
                        "is_final": False,  # 🔧 标记为常规摘要
                        "sentence_count": len(snapshot),  # 兼容字段
                        "duration": duration  # 兼容字段
                    }
                    
                    # 🔧 详细日志
                    print(f"📤 Sending summary to client:")
                    print(f"   ID: {summary_data['id']}")
                    print(f"   Is Final: {summary_data['is_final']}")
                    print(f"   Created At: {summary_data['created_at']}")
                    print(f"   Content length: {len(summary_data['content'])} chars")
                    print(f"   JSON preview: {json.dumps(summary_data)[:200]}...")
                    
                    await session.websocket.send_text(
                        f"[summary] {json.dumps(summary_data)}"
                    )
                    
                    print(f"📤 Sent summary to client")
                    
                except Exception as e:
                    print(f"⚠️ Failed to send summary to client: {e}")
                # 🔧 ==================== 发送结束 ====================
                
            except Exception as e:
                print(f"❌ Database save failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ==================== Phase 5: 更新上下文缓存 ====================
        self._update_context_cache(session, snapshot)
        
        # ==================== Phase 6: 清理状态 ====================
        self._cleanup_after_summary(session, snapshot)
        
        print(f"\n{'='*60}")
        print(f"✅ SUMMARY GENERATION COMPLETE")
        print(f"{'='*60}\n")
        
    def _build_elevenlabs_config_for_mode(self, mode: str) -> ElevenLabsConfig:
        """
        Build an ElevenLabsConfig for the given mode.
        
        Now uses centralized configuration from config.py.
        
        Args:
            mode: "lecture" or "discussion"
        
        Returns:
            ElevenLabsConfig with mode-specific settings
        """
        # use centralized config
        mode_config = TranscriptionConfig.get_mode_config(mode)
        
        return ElevenLabsConfig(
            # Common settings
            audio_format=mode_config.audio_format,
            sample_rate=mode_config.sample_rate,
            language_code=mode_config.language_code,
            timestamps_granularity=mode_config.timestamps_granularity,
            model_id=mode_config.model_id,
            mode=mode,
            
            # Commit strategy
            commit_strategy=mode_config.commit_strategy,
            
            # VAD settings (None for manual mode)
            vad_silence_threshold_secs=mode_config.vad_silence_threshold_secs,
            vad_threshold=mode_config.vad_threshold,
            min_speech_duration_ms=mode_config.min_speech_duration_ms,
            min_silence_duration_ms=mode_config.min_silence_duration_ms,
        )

    async def create_session(self, websocket: WebSocket) -> LiveSession:
        """
        Create a new session for the given WebSocket connection.

        - Creates the Session object with basic metadata.
        - ElevenLabs client will be created lazily when MODE is received.
        - This allows users to connect without immediately consuming resources.
        """
        session_id = str(uuid4())
        session = LiveSession(
            id=session_id,
            websocket=websocket,
            meta={},  
            started_at=datetime.now()  
        )
        self.sessions[session_id] = session
        print(f"[SessionManager] ✅ Session created: {session_id}")
        print(f"   Ingestion buffer: {len(session.ingestion_buffer)}")
        print(f"   Context cache: {len(session.context_cache)}")
        print(f"   Is generating: {session.is_generating}")
        
        # Log session creation
        if self.run_logger is not None:
            self.run_logger.log_event(session_id, {"type": "session_created"})
        
        return session

    async def close_session(self, session_id: str) -> None:
        """
        Close the session with the given ID.

        Steps:
        1. Mark session as inactive
        2. Cancel manual commit task if running
        3. Close ElevenLabs client connection
        4. Close WebSocket (if not already closed by client)
        5. Notify RunLogger
        6. Remove from sessions dictionary
        
        Note: Does NOT auto-save session data - user must explicitly call SAVE
        """
        session = self.sessions.get(session_id)
        if not session:
            print(f"[SessionManager] Session {session_id} not found, already closed?")
            return

        print(f"[SessionManager] Closing session {session_id}")
    
        
        # Mark inactive
        session.is_active = False

        # Cancel manual commit task if running
        if session.manual_commit_task is not None:
            session.manual_commit_task.cancel()
            try:
                await session.manual_commit_task
            except asyncio.CancelledError:
                pass
            print(f"[SessionManager] Cancelled manual commit task for {session_id}")

        # Close ElevenLabs client
        if session.eleven_client is not None:
            try:
                await session.eleven_client.close()
                print(f"[SessionManager] Closed ElevenLabs client for {session_id}")
            except Exception as exc:
                print(f"[SessionManager] Error closing ElevenLabs client: {exc}")

        # Close WebSocket if still open
        try:
            # Check if WebSocket is still connected
            if hasattr(session.websocket, 'client_state'):
                from fastapi.websockets import WebSocketState
                if session.websocket.client_state != WebSocketState.DISCONNECTED:
                    await session.websocket.close(code=1000, reason="Session closed normally")
            else:
                await session.websocket.close()
            print(f"[SessionManager] Closed WebSocket for {session_id}")
        except Exception as exc:
            print(f"[SessionManager] WebSocket already closed or error: {exc}")

        # Finish run logging
        if self.run_logger is not None:
            self.run_logger.finish_run(session_id)

        # Remove session from dictionary
        del self.sessions[session_id]
        print(f"[SessionManager] Session {session_id} fully cleaned up")
    
    def _should_generate_summary(self, session: LiveSession) -> bool:
        """
        检查是否应该生成摘要
        
        条件：
        1. 距离上次摘要已超过配置的时间间隔
        2. 累积了足够数量的句子
        3. 会话仍然活跃
        4. 有关联的数据库会话ID
        5. (可选) 语义完整性检查
        """
        # 🔧 使用配置类
        interval = SummaryConfig.SUMMARY_INTERVAL_SECONDS
        min_sentences = SummaryConfig.MIN_SENTENCES
        loose_threshold = SummaryConfig.LOOSE_MODE_THRESHOLD
        
        now = time.time()
        elapsed = now - session.last_summary_time
        
        # 🔧 添加：总是打印检查状态（调试用）
        print(f"\n🔍 TRIGGER CHECK:")
        print(f"   ⏱️  Time: {elapsed:.1f}s / {interval}s (need: {interval}s)")
        print(f"   📝 Buffer: {len(session.ingestion_buffer)} / {min_sentences} (need: {min_sentences})")
        print(f"   ✅ Active: {session.is_active}")
        print(f"   💾 DB session: {session.db_session_id is not None}")
        print(f"   🔒 Generating: {session.is_generating}")
        
        # 基础条件检查
        has_time = elapsed >= interval
        has_sentences = len(session.ingestion_buffer) >= min_sentences
        is_active = session.is_active
        has_db_session = session.db_session_id is not None
        
        # 如果基础条件不满足，直接返回
        if not (has_time and has_sentences and is_active and has_db_session):
            print(f"   ❌ Basic conditions not met:")
            if not has_time:
                print(f"      - Time not enough: {elapsed:.1f}s < {interval}s")
            if not has_sentences:
                print(f"      - Not enough sentences: {len(session.ingestion_buffer)} < {min_sentences}")
            if not is_active:
                print(f"      - Session not active")
            if not has_db_session:
                print(f"      - No DB session")
            print()  # 空行
            return False
        
        # 🔧 基础条件已满足，检查语义完整性
        print(f"   ✅ Basic conditions met!")
        
        # 如果超过宽松模式阈值，忽略语义检查
        if elapsed >= loose_threshold:
            print(f"   ⏰ Loose mode: {elapsed:.1f}s >= {loose_threshold}s")
            print(f"   ✅ Bypassing semantic check, triggering immediately\n")
            return True
        
        # 标准模式：检查最后一句是否完整
        if session.ingestion_buffer:
            last_sentence = session.ingestion_buffer[-1]
            has_ender = self._has_sentence_ender(last_sentence)
            
            if has_ender:
                print(f"   ✅ Semantic check passed: sentence ends with proper ender")
                print(f"      Last: '{last_sentence[:50]}{'...' if len(last_sentence) > 50 else ''}'")
                print()  # 空行
                return True
            else:
                print(f"   ⏳ Waiting for sentence completion")
                print(f"      Last: '{last_sentence[:50]}{'...' if len(last_sentence) > 50 else ''}'")
                print()  # 空行
                return False
        
        print(f"   ⚠️ Ingestion buffer is empty\n")
        return False
              
    async def _save_session_to_db(self, session_id: str) -> None:
        """
        save the current session's transcript to the database.
        """
        session = self.sessions.get(session_id)
        if session is None:
            print(f"[SessionManager] Cannot save: session {session_id} not found")
            return
        
        # check if already saved
        if session.is_saved:
            print(f"[SessionManager] LiveSession {session_id} already saved")
            try:
                await session.websocket.send_text("[save] Session already saved")
            except:
                pass
            return
        
        # check if db_session_id exists
        if session.db_session_id is None:
            print(f"[SessionManager] Cannot save: no db_session_id for session {session_id}")
            try:
                await session.websocket.send_text("[save] ERROR: No database session")
            except:
                pass
            return
        
        # check if there is any transcript to save
        if not session.transcript_parts:
            print(f"[SessionManager] Cannot save: no transcript for session {session_id}")
            try:
                await session.websocket.send_text("[save] ERROR: No transcript to save")
            except:
                pass
            return
        
        # save to database
        try:
            from database.db import DatabaseManager
            from datetime import datetime
            
            # calculate duration
            duration = int(time.time() - session.start_time)
            
            # combine transcript parts
            full_transcript = "\n".join(session.transcript_parts)
            
            # statistics
            sentence_count = len(session.transcript_parts)
            char_count = len(full_transcript)
            
            # renew session record in database
            DatabaseManager.update_session(
                session_id=session.db_session_id,
                duration_seconds=duration,
                transcript_text=full_transcript,
                sentence_count=sentence_count,
                char_count=char_count,
                ended_at=datetime.utcnow()
            )
            
            # mark as saved
            session.is_saved = True

            print(f"[SessionManager] ✅ Saved session to DB")
            print(f"   DB Session ID: {session.db_session_id}")
            print(f"   Duration: {duration}s")
            print(f"   Sentences: {sentence_count}")
            print(f"   Characters: {char_count}")
            print(f"   Summaries: {len(session.generated_summaries)}")

            # save to local file (if project_id available)
            project_id = session.meta.get("project_id")
            if project_id is not None:
                try:
                    # fetch project info
                    project = DatabaseManager.get_project_by_id(project_id)
                    if project:
                        # determine mode
                        mode = session.meta.get("mode", "unknown")
                        
                        # save to file
                        filepath = self._save_transcript_to_file(
                            project_name=str(project.name),
                            mode=mode,
                            transcript_text=full_transcript,
                            started_at=session.started_at.isoformat() if session and session.started_at else datetime.now().isoformat()
                        )
                        
                        if filepath:
                            print(f"[SessionManager] 📄 File saved: {filepath}")
                    else:
                        print(f"[SessionManager] ⚠️ Project {project_id} not found for file export")
                except Exception as exc:
                    print(f"[SessionManager] ❌ Failed to export to file: {exc}")

            # notify client - save complete
            try:
                save_message = json.dumps({
                    "status": "saved",
                    "session_id": session.db_session_id,
                    "duration": duration,
                    "sentences": sentence_count,
                    "characters": char_count,
                    "summaries": len(session.generated_summaries)
                })
                await session.websocket.send_text(f"[save_complete] {save_message}")
                print(f"[SessionManager] 📤 Sent [save_complete] to client")
            except Exception as exc:
                print(f"[SessionManager] Failed to send save_complete: {exc}")
            
            # 🔧 ==================== 新增：触发索引 ====================
            
            # 检查是否有摘要需要索引
            if session.db_session_id and len(session.generated_summaries) > 0:
                print(f"\n📇 Triggering indexing for session {session.db_session_id}")
                
                # 发送索引开始通知到前端
                try:
                    await session.websocket.send_text("[indexing_start]")
                    print("📤 Sent [indexing_start] to client")
                except Exception as e:
                    print(f"⚠️ Failed to send indexing_start: {e}")
                
                # 异步触发索引任务（不阻塞保存操作）
                asyncio.create_task(
                    self._index_session_async(session.db_session_id, session.websocket)
                )
            else:
                if len(session.generated_summaries) == 0:
                    print("ℹ️ No summaries to index")
                else:
                    print("⚠️ Cannot index: no db_session_id")
            
            # 🔧 ==================== 新增结束 ====================
            
        except Exception as exc:
            print(f"[SessionManager] ❌ Failed to save session to DB: {exc}")
            import traceback
            traceback.print_exc()
            
            try:
                await session.websocket.send_text(f"[save] ERROR: {str(exc)}")
            except:
                pass
    
    async def _index_session_async(self, db_session_id: int, websocket: WebSocket):
        """
        异步索引会话摘要
        
        Args:
            db_session_id: 数据库 session ID
            websocket: WebSocket 连接（用于发送完成通知）
        """
        try:
            print(f"\n{'='*60}")
            print(f"📇 Starting async indexing for session {db_session_id}")
            print(f"{'='*60}\n")
            
            # 调用索引服务
            result = await self.indexing_service.index_session_summaries(db_session_id)
            
            if result["success"]:
                # 发送索引完成通知
                message = json.dumps({
                    "status": "completed",
                    "session_id": db_session_id,
                    "indexed_count": result.get("indexed", 0),
                    "project_id": result.get("project_id")
                })
                
                try:
                    await websocket.send_text(f"[indexing_complete] {message}")
                    print(f"📤 Sent [indexing_complete] to client")
                    print(f"   Indexed: {result.get('indexed', 0)} summaries")
                except Exception as e:
                    print(f"⚠️ Failed to send indexing_complete: {e}")
            else:
                # 发送索引失败通知
                error_message = json.dumps({
                    "status": "failed",
                    "session_id": db_session_id,
                    "error": result.get("error", "Unknown error")
                })
                
                try:
                    await websocket.send_text(f"[indexing_error] {error_message}")
                    print(f"❌ Indexing failed: {result.get('error')}")
                except Exception as e:
                    print(f"⚠️ Failed to send indexing_error: {e}")
            
        except Exception as e:
            print(f"\n❌ Async indexing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试发送错误通知
            try:
                error_message = json.dumps({
                    "status": "failed",
                    "session_id": db_session_id,
                    "error": str(e)
                })
                await websocket.send_text(f"[indexing_error] {error_message}")
            except:
                pass
            
    def _save_transcript_to_file(
        self,
        project_name: str,
        mode: str,
        transcript_text: str,
        started_at: str) -> Optional[str]:
        """
        Save the transcript to /transcripts.
        
        Args:
            project_name: project name for directory
            mode:  (lecture/discussion)
            transcript_text: full transcript content
            started_at: start time in ISO format string
        
        Returns:
            The file path if saved successfully, else None.
        """
        try:
            from pathlib import Path
            from datetime import datetime
            
            # Base directory            
            base_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "transcripts"


            safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
            project_dir = base_dir / safe_project_name
            
            # create directories if not exist
            project_dir.mkdir(parents=True, exist_ok=True)

            
            # filename with timestamp
            # eg. 2025-12-01_14-30-15_lecture.txt
            try:
                dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
            
            timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_{mode}.txt"
            filepath = project_dir / filename
            
            # write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                # write metadata header
                f.write(f"Project: {project_name}\n")
                f.write(f"Mode: {mode}\n")
                f.write(f"Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 60 + "\n\n")
                
                # write transcript
                f.write(transcript_text)
            
            print(f"[SessionManager] 📄 Transcript saved to file: {filepath}")
            return str(filepath)
            
        except Exception as exc:
            print(f"[SessionManager] ❌ Failed to save transcript to file: {exc}")
            import traceback
            traceback.print_exc()
            return None

    async def handle_binary_audio(self, session_id: str, data: bytes) -> None:
        """Handle a binary message (audio data) from the frontend client."""
        session = self.sessions.get(session_id)
        
        # 🔇 条件日志：只在详细模式打印音频块
        if LogConfig.LOG_AUDIO_CHUNKS:
            print(f"[SessionManager] 🎵 Audio chunk: {len(data)} bytes")
        
        if session is None or not session.is_active:
            if LogConfig.VERBOSE:
                print(f"[SessionManager] ⚠️ Audio for invalid/inactive session {session_id}")
            return

        if session.eleven_client is None:
            print(f"[SessionManager] ❌ Audio received but ElevenLabs client not initialized")
            print(f"[SessionManager] Client should send MODE: first")
            return

        # Record last audio time
        session.last_audio_time = time.time()

        # Check and reconnect if needed
        if not session.eleven_client.is_alive():
            print(f"[SessionManager] ⚠️ ElevenLabs connection is dead, attempting to reconnect...")
            
            idle_time = time.time() - session.last_audio_time
            print(f"[SessionManager] Connection was idle for {idle_time:.1f}s")
            
            try:
                await session.eleven_client.close()
                await session.eleven_client.connect()
                print(f"[SessionManager] ✅ Reconnected to ElevenLabs")
                
                # 🔧 重连后重启 manual commit 任务
                mode = session.meta.get("mode", "lecture")
                mode_config = TranscriptionConfig.get_mode_config(mode)
                
                if mode_config.commit_strategy == "manual":
                    # Cancel old task
                    if session.manual_commit_task is not None:
                        session.manual_commit_task.cancel()
                        try:
                            await session.manual_commit_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Start new task
                    commit_interval = mode_config.commit_interval
                    assert commit_interval is not None, \
                        f"Manual mode requires commit_interval, but got None for mode '{mode}'"
                    
                    session.manual_commit_task = asyncio.create_task(
                        self._manual_commit_loop(session_id, commit_interval)
                    )
                    
                    print(f"[SessionManager] Restarted manual commit loop after reconnection")
                
            except Exception as e:
                error_msg = f"Failed to reconnect to ElevenLabs: {str(e)}"
                print(f"[SessionManager] ❌ {error_msg}")
                
                try:
                    await session.websocket.send_text(f"[error] {error_msg}")
                except Exception:
                    pass
                return

        # 🔇 删除：这行重复的日志已被移除
        # print(f"[SessionManager] Audio chunk from {session_id}: {len(data)} bytes")

        # Forward to ElevenLabs
        try:
            await session.eleven_client.send_audio_chunk(data)
        except Exception as exc:
            error_msg = f"Failed to send audio to ElevenLabs: {str(exc)}"
            print(f"[SessionManager] ❌ {error_msg}")
            
            try:
                await session.websocket.send_text(f"[error] {error_msg}")
            except Exception:
                pass

        # 🔇 条件日志：只在启用时记录到 RunLogger
        if self.run_logger is not None and LogConfig.LOG_RUN_EVENTS:
            self.run_logger.log_event(
                session_id,
                {"type": "audio_chunk", "size": len(data)}
            )
    
    async def handle_text_message(self, session_id: str, text: str) -> None:
        """Handle a text message coming from the frontend client."""
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            print(f"[SessionManager] Received text for invalid/inactive session {session_id}")
            return

        stripped = text.strip()
        
        # ==================== PROJECT_ID Configuration ====================
        # 使用 PROJECT_ID 
        if stripped.upper().startswith("PROJECT_ID:"):
            try:
                project_id_str = stripped.split(":", 1)[1].strip()
                project_id = int(project_id_str)
                session.meta["project_id"] = project_id
                
                print(f"[SessionManager] ✅ Set project_id={project_id} for session {session_id}")
                print(f"   Current meta: {session.meta}")
                
                try:
                    await session.websocket.send_text(f"[config] project_id={project_id}")
                except Exception as exc:
                    print(f"[SessionManager] Failed to send project acknowledgment: {exc}")
                
                return
                
            except (ValueError, IndexError) as exc:
                print(f"[SessionManager] ❌ Invalid PROJECT_ID message: {text}, error: {exc}")
                try:
                    await session.websocket.send_text(f"[error] Invalid project_id")
                except Exception:
                    pass
                return

        # ==================== MODE Configuration ====================
        if stripped.upper().startswith("MODE:"):
            mode_raw = stripped.split(":", 1)[1].strip().lower()
            mode = "lecture" if mode_raw not in ["discussion"] else "discussion"
            
            # 🔧 添加调试日志
            print(f"\n{'='*60}")
            print(f"🔧 MODE COMMAND RECEIVED")
            print(f"   Session ID: {session_id}")
            print(f"   Mode: {mode}")
            print(f"   Current meta: {session.meta}")
            print(f"   Has project_id: {'project_id' in session.meta}")
            if 'project_id' in session.meta:
                print(f"   Project ID value: {session.meta['project_id']} (type: {type(session.meta['project_id'])})")
            print(f"{'='*60}\n")
            
            session.meta["mode"] = mode

            # Skip if client already exists
            if session.eleven_client is not None:
                print(f"[SessionManager] Mode already configured for {session_id}, skipping")
                try:
                    await session.websocket.send_text(f"[config] mode already set to {mode}")
                except Exception as exc:
                    print(f"[SessionManager] Failed to send acknowledgment: {exc}")
                return

            # Build config and connect
            print(f"[SessionManager] Setting mode to '{mode}' for session {session_id}")
            
            if self.run_logger is not None:
                self.run_logger.log_event(session_id, {"type": "mode_set", "mode": mode})

            try:
                config = self._build_elevenlabs_config_for_mode(mode)
                eleven_client = ElevenLabsRealtimeClient(
                api_key=self._api_key,  
                config=config
            )

                # Setup callbacks
                def on_partial(text: str):
                    asyncio.create_task(
                        self.push_transcript_to_client(session_id, text, is_final=False)
                    )

                def on_final(text: str):
                    asyncio.create_task(
                        self.push_transcript_to_client(session_id, text, is_final=True)
                    )

                eleven_client.on_partial = on_partial
                eleven_client.on_final = on_final

                # Connect
                await eleven_client.connect()
                session.eleven_client = eleven_client
                success_msg = f"[config] mode set to {mode}, connected to ElevenLabs"
                print(f"[SessionManager] {success_msg}")
                
                # ==================== 数据库 Session 管理 ====================
                try:
                    from database.db import DatabaseManager
                    
                    # 🔧 获取 project_id
                    project_id = session.meta.get("project_id")
                    
                    if project_id is None:
                        print(f"[SessionManager] ❌ No project_id in session.meta")
                        print(f"   Current meta: {session.meta}")
                        print(f"   Cannot create DB session without project_id")
                        try:
                            await session.websocket.send_text("[error] No project selected. Please select a project first.")
                        except Exception:
                            pass
                        # ⚠️ 不 return，继续启动转录（但不保存到数据库）
                    else:
                        print(f"[SessionManager] 📋 Project ID: {project_id}")
                        
                        # 🔧 检查是否复用现有 session
                        if session.db_session_id is not None and not session.is_saved:
                            # 复用现有未保存的 session
                            print(f"[SessionManager] 📝 Reusing existing DB session: {session.db_session_id}")
                            
                            # Phase 2: 重置摘要相关状态
                            print(f"[SessionManager] 🔄 Resetting summary state for reused session")
                            session.last_summary_time = time.time()
                            session.ingestion_buffer.clear()
                            session.processing_snapshot.clear()
                            session.context_cache.clear()
                            session.sentence_count_since_last = 0
                            session.is_generating = False
                            
                            # 向后兼容：清空旧字段
                            session.summary_buffer.clear()
                            session.last_context = ""
                            session.generated_summaries.clear()
                            
                            print(f"   ✅ Ingestion buffer: {len(session.ingestion_buffer)}")
                            print(f"   ✅ Context cache: {len(session.context_cache)}")
                            print(f"   ✅ Is generating: {session.is_generating}")
                            
                        else:
                            # 创建新的数据库 session
                            db_session = DatabaseManager.create_session(
                                project_id=project_id,
                                mode=mode
                            )
                            session.db_session_id = db_session.id
                            session.is_saved = False
                            
                            # Phase 2: 初始化摘要状态
                            session.last_summary_time = time.time()
                            session.ingestion_buffer.clear()
                            session.processing_snapshot.clear()
                            session.context_cache.clear()
                            session.sentence_count_since_last = 0
                            session.is_generating = False
                            session.generated_summaries.clear()
                            
                            # 清空转录（新会话）
                            session.transcript_parts.clear()
                            session.summary_buffer.clear()
                            session.last_context = ""
                            
                            print(f"[SessionManager] ✅ Created new DB session: {db_session.id} for project {project_id}")
                            print(f"   Mode: {mode}")
                            print(f"   Summary state initialized")
                            
                except Exception as exc:
                    print(f"[SessionManager] ❌ Failed to create/reuse DB session: {exc}")
                    import traceback
                    traceback.print_exc()

                # ==================== Manual Commit Loop ====================
                if config.commit_strategy == "manual":
                    mode_config = TranscriptionConfig.get_mode_config(mode)
                    commit_interval = mode_config.commit_interval
                    assert commit_interval is not None, \
                        f"Manual mode requires commit_interval, but got None for mode '{mode}'"
                    
                    session.manual_commit_task = asyncio.create_task(
                        self._manual_commit_loop(session_id, commit_interval)
                    )
                    print(f"[SessionManager] Started manual commit loop with {commit_interval}s interval")

            except Exception as exc:
                error_msg = f"[error] Failed to connect to ElevenLabs: {str(exc)}"
                print(f"[SessionManager] {error_msg}")
                session.meta["connection_error"] = str(exc)
                
                try:
                    await session.websocket.send_text(error_msg)
                except Exception:
                    pass
                return

            return

        # ==================== STOP Signal ====================
        if stripped.upper() == "STOP":
                    print(f"[SessionManager] Received STOP signal from {session_id}")
                    
                    # 🔧 ==================== Step 1: 生成最终摘要 ====================
                    
                    # 检查是否有转录内容
                    full_transcript = "\n".join(session.transcript_parts)
                    
                    if full_transcript and len(full_transcript.strip()) > 20 and session.db_session_id is not None:
                        print(f"\n{'='*60}")
                        print(f"🏁 GENERATING FINAL SUMMARY")
                        print(f"   Session: {session_id}")
                        print(f"   Total sentences: {len(session.transcript_parts)}")
                        print(f"   Total characters: {len(full_transcript)}")
                        print(f"{'='*60}\n")
                        
                        # 发送最终摘要开始信号
                        try:
                            await session.websocket.send_text("[final_summary_start]")
                        except Exception as e:
                            print(f"⚠️ Failed to send final_summary_start: {e}")
                        
                        # 生成最终摘要
                        try:
                            await self._generate_final_summary(session)
                        except Exception as e:
                            print(f"❌ Failed to generate final summary: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        if len(full_transcript.strip()) <= 20:
                            print(f"ℹ️  Transcript too short for final summary ({len(full_transcript)} chars)")
                        elif session.db_session_id is None:
                            print(f"ℹ️  No database session, skipping final summary")
                    
                    # 🔧 ==================== Step 2: 清理资源 ====================
                    
                    # 取消手动提交任务
                    if session.manual_commit_task is not None:
                        session.manual_commit_task.cancel()
                        session.manual_commit_task = None
                        print(f"[SessionManager] Cancelled manual commit task for {session_id}")
                    
                    # 关闭 ElevenLabs 客户端
                    if session.eleven_client is not None:
                        await session.eleven_client.close()
                        session.eleven_client = None
                        print(f"[SessionManager] Closed ElevenLabs client for {session_id}")
                    
                    # 🔧 ==================== Step 3: 发送确认消息 ====================
                    
                    try:
                        await session.websocket.send_text("[config] recording stopped, session kept alive")
                    except Exception as exc:
                        print(f"[SessionManager] Failed to send stop acknowledgment: {exc}")

                    return
        
        # ==================== SAVE Signal ====================
        if stripped.upper() == "SAVE":
            print(f"[SessionManager] Received SAVE for session {session_id}")
            await self._save_session_to_db(session_id)
            return

        # ==================== Other Text (Echo) ====================
        print(f"[SessionManager] Text from {session_id}: {text}")
        echo_text = f"Echo: {text}"

        try:
            await session.websocket.send_text(echo_text)
        except Exception as exc:
            print(f"[SessionManager] Failed to send echo to {session_id}: {exc}")

        if self.run_logger is not None:
            self.run_logger.log_event(
                session_id,
                {"type": "text_echo", "received": text, "sent": echo_text}
            )
    
    async def _generate_final_summary(self, session: LiveSession) -> None:
        """
        生成最终摘要（Stop 时调用）
        
        使用完整的 full transcript 生成一个总结性摘要
        
        Args:
            session: LiveSession 对象
        """
        try:
            # 🔧 获取完整转录
            full_transcript = "\n".join(session.transcript_parts)
            
            if not full_transcript or len(full_transcript.strip()) < 20:
                print(f"⚠️  Transcript too short, skipping final summary")
                return
            
            print(f"📝 Generating final summary from full transcript")
            print(f"   Total sentences: {len(session.transcript_parts)}")
            print(f"   Total characters: {len(full_transcript)}")
            
            # 🔧 获取模式
            mode = session.meta.get("mode", "discussion")
            
            print(f"🤖 Calling SummaryService for final summary...")
            print(f"   Mode: {mode}")
            
            # 🔧 调用 SummaryService 生成摘要
            from summary_service import SummaryService
            summary_service = SummaryService()
            
            # 🔧 使用实际的方法签名
            summary_content = await summary_service.generate_summary(
                buffer_text=full_transcript,  # 🔧 使用 buffer_text 参数
                context="",  # 🔧 最终摘要不需要上下文
                mode=mode
            )
            
            if not summary_content or len(summary_content.strip()) == 0:
                print(f"⚠️  Empty summary generated, skipping")
                return
            
            print(f"✅ Final summary generated successfully")
            print(f"   Output length: {len(summary_content)} chars")
            print(f"   Preview: {summary_content[:150]}...")
            
            # 🔧 保存到数据库
            if session.db_session_id:
                try:
                    from database.db import DatabaseManager
                    
                    project_id = session.meta.get("project_id")
                    
                    summary = DatabaseManager.create_summary(
                        session_id=session.db_session_id,
                        project_id=project_id,
                        content=summary_content,
                        source_text=full_transcript[:1000],  # 保存前1000字符作为来源
                        start_sentence_idx=0,
                        end_sentence_idx=len(session.transcript_parts),
                        duration_seconds=int(time.time() - session.start_time) if hasattr(session, 'start_time') else 0
                    )
                    
                    print(f"✅ [DB] Saved final summary: id={summary.id}, session={session.db_session_id}")
                    
                    # 🔧 发送给前端
                    try:
                        summary_data = {
                            "id": summary.id,
                            "content": summary_content,
                            "created_at": summary.created_at.isoformat() if summary.created_at else "",
                            "is_final": True  # 🔧 标记为最终摘要
                        }
                        
                        await session.websocket.send_text(
                            f"[summary] {json.dumps(summary_data)}"
                        )
                        
                        print(f"📤 Sent final summary to client")
                        
                    except Exception as e:
                        print(f"⚠️ Failed to send final summary to client: {e}")
                    
                except Exception as e:
                    print(f"❌ Failed to save final summary to database: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ No database session ID, cannot save final summary")
            
        except Exception as e:
            print(f"❌ Error in _generate_final_summary: {e}")
            import traceback
            traceback.print_exc()
            
    async def push_transcript_to_client(
        self,
        session_id: str,
        text: str,
        is_final: bool,
    ) -> None:
        """
        Send transcription text back to the frontend client.

        Called from ElevenLabs client callbacks (on_partial / on_final).

        Message format: "[partial] text" or "[final] text"
        - Frontend can parse this to update current subtitle vs full transcript
        
        Args:
            session_id: Which session should receive the text.
            text: The transcript content.
            is_final: True for committed transcript, False for partial.
        """
        # 🔧 获取 session
        session = self.sessions.get(session_id)
        if session is None or not session.is_active:
            if LogConfig.VERBOSE:
                print(f"[SessionManager] Cannot push transcript, session {session_id} is invalid/inactive")
            return
        # 🔧 构建消息
        msg_type = "final" if is_final else "partial"
        payload = f"[{msg_type}] {text}"
        
        # 🔧 Phase 2: 处理 final 转录（使用新的双容器架构）
        if is_final and text.strip():
            stripped_text = text.strip()
            
            # 添加到完整转录
            session.transcript_parts.append(stripped_text)
            
            # 🔧 添加到接收缓冲区（新逻辑）
            session.ingestion_buffer.append(stripped_text)
            session.sentence_count_since_last += 1
            
            if LogConfig.LOG_TRANSCRIPT:
                print(f"📥 [final] '{stripped_text[:60]}{'...' if len(stripped_text) > 60 else ''}'")
                print(f"   Buffer: {len(session.ingestion_buffer)} | Count: {session.sentence_count_since_last}")
            
            # 检查是否应该生成摘要
            should_trigger = self._should_generate_summary(session)
            
            if should_trigger:
                if LogConfig.LOG_SUMMARY:
                    print(f"🎯 Trigger condition met, initiating summary generation")
                asyncio.create_task(self._generate_and_send_summary(session))
            else:
                if LogConfig.VERBOSE:
                    elapsed = time.time() - session.last_summary_time
                    print(f"⏳ Waiting: {elapsed:.1f}s, {len(session.ingestion_buffer)} sentences")
            
            # 🔇 只在 VERBOSE 模式显示状态快照
            if LogConfig.VERBOSE:
                print(f"📊 State: ingestion={len(session.ingestion_buffer)}, "
                    f"snapshot={len(session.processing_snapshot)}, "
                    f"cache={len(session.context_cache)}, "
                    f"generating={session.is_generating}")

        # 发送消息到前端
        try:
            await session.websocket.send_text(payload)
            
            # 🔇 只在 VERBOSE 模式记录发送
            if LogConfig.LOG_WEBSOCKET_MESSAGES:
                print(f"[SessionManager] ✅ Pushed {msg_type} to {session_id}: {text[:50]}...")
                
        except Exception as exc:
            print(f"[SessionManager] ❌ Failed to push transcript to {session_id}: {exc}")
            import traceback
            traceback.print_exc()
            return

        # 记录事件到 RunLogger
        if self.run_logger is not None and LogConfig.LOG_RUN_EVENTS:
            try:
                self.run_logger.log_event(
                    session_id,
                    {"type": "transcript", "is_final": is_final, "text": text}
                )
            except Exception as exc:
                if LogConfig.VERBOSE:
                    print(f"[SessionManager] ⚠️ Failed to log event: {exc}")

    async def _manual_commit_loop(self, session_id: str, interval: float) -> None:
        """
        Background task for manual commit strategy.

        For discussion/presentation mode, automatically commits transcripts
        at regular intervals (default 12s based on Experiment C).

        This loop runs until the session is closed or the task is cancelled.
        """
        print(f"[SessionManager] Manual commit loop started for {session_id} (interval={interval}s)")
        
        try:
            while True:
                await asyncio.sleep(interval)
                
                session = self.sessions.get(session_id)
                if session is None or not session.is_active:
                    print(f"[SessionManager] Manual commit loop: session {session_id} no longer active")
                    break

                if session.eleven_client is not None:
                    # 🔧 新增：检查连接
                    if not session.eleven_client.is_alive():
                        print(f"[SessionManager] ⚠️ Skipping commit - connection not alive for {session_id}")
                        continue
                    
                    try:
                        # 🔧 修改：实际发送 commit
                        await session.eleven_client.send_commit()
                        print(f"[SessionManager] ✅ Manual commit sent for {session_id}")
                        
                        if self.run_logger is not None:
                            self.run_logger.log_event(
                                session_id,
                                {"type": "manual_commit", "interval": interval}
                            )
                        
                    except Exception as exc:
                        print(f"[SessionManager] ❌ Error sending manual commit: {exc}")

        except asyncio.CancelledError:
            print(f"[SessionManager] Manual commit loop cancelled for {session_id}")
            raise
        except Exception as exc:
            print(f"[SessionManager] Unexpected error in manual commit loop: {exc}")
            
    def _cleanup_after_summary(self, session: LiveSession, snapshot: List[str]) -> None:
        """
        清理摘要生成后的状态
        
        Args:
            session: LiveSession 对象
            snapshot: 刚处理的句子快照
        """
        # 清空 processing_snapshot
        session.processing_snapshot.clear()
        
        # 重置生成状态
        session.is_generating = False
        
        # 更新最后生成摘要的时间
        session.last_summary_time = time.time()
        
        print(f"✅ State updated:")
        print(f"   Context cache: {len(session.context_cache)} sentences")
        print(f"   Next start index: {session.next_start_sentence_idx}")
        print(f"   Is generating: {session.is_generating}")
        print(f"   Generated summaries: {len(session.generated_summaries)}")
   