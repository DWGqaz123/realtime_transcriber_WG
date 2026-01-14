# backend/main.py

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import sys

if getattr(sys, 'frozen', False):
    # 打包环境
    log_file = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "backend.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 重定向 stdout 和 stderr
    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout
    
    print(f"🔧 Backend log started: {log_file}")

# ==================== 加载环境变量 ====================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# ==================== 导入配置和服务 ====================

from config import LogConfig, SummaryConfig, validate_api_keys, ELEVENLABS_API_KEY, OPENAI_API_KEY

# ==================== 导入数据库 ====================

try:
    from database.db import DatabaseManager
    print("✅ DatabaseManager imported successfully")
except Exception as e:
    print(f"❌ Failed to import DatabaseManager: {e}")
    import traceback
    traceback.print_exc()
    DatabaseManager = None

# ==================== 导入 RunLogger ====================

try:
    from run_logger import RunLogger
    print("✅ RunLogger imported successfully")
except Exception as e:
    print(f"❌ Failed to import RunLogger: {e}")
    import traceback
    traceback.print_exc()
    RunLogger = None

# ==================== 导入 SessionManager ====================

try:
    from session_manager import SessionManager
    print("✅ SessionManager imported successfully")
except Exception as e:
    print(f"❌ Failed to import SessionManager: {e}")
    import traceback
    traceback.print_exc()
    SessionManager = None

# ==================== 导入路由 ====================

try:
    from routes import projects, search
    print("✅ Projects router imported successfully")
except Exception as e:
    print(f"❌ Failed to import routers: {e}")
    import traceback
    traceback.print_exc()

print(f"📊 After imports: SessionManager={SessionManager}, RunLogger={RunLogger}")

# ==================== 创建 FastAPI 应用 ====================

app = FastAPI(
    title="Realtime Transcriber API",
    description="API for real-time transcription with project management",
    version="2.0.0"
)

print("✅ FastAPI app created")

# ==================== CORS 配置 ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 注册路由 ====================

app.include_router(projects.router)
print("✅ Projects router registered")
app.include_router(search.router)

# ==================== 全局变量 ====================

session_manager: Optional[SessionManager] = None
run_logger: Optional[RunLogger] = None
api_keys_valid: bool = False

# ==================== 验证 API Keys ====================

print("\n" + "="*60)
print("🚀 STEP 4: Initializing Backend")
print("="*60)

api_keys_valid = validate_api_keys()

if not api_keys_valid:
    print("\n⚠️  Backend will start but functionality may be limited without API Keys.")
    print("Please configure API Keys in the app Settings and restart.\n")

# ==================== 基础端点 ====================

@app.get("/")
async def root():
    """API 根端点"""
    return {
        "message": "Realtime Transcriber API",
        "version": "2.0.0",
        "status": "running" if session_manager is not None else "limited",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "projects": "/api/projects",
            "websocket": "/ws/transcribe"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok" if session_manager is not None else "degraded",
        "api_keys_configured": api_keys_valid,
        "session_manager_ready": session_manager is not None
    }

@app.get("/api/status/keys")
async def check_api_keys():
    """检查 API Keys 配置状态"""
    return {
        "openai_configured": bool(OPENAI_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "all_configured": bool(OPENAI_API_KEY and ELEVENLABS_API_KEY)
    }

# ==================== WebSocket 端点 ====================

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """实时转录 WebSocket 端点"""
    
    # 🔧 检查 session_manager 是否已初始化
    if session_manager is None:
        await websocket.close(code=1011, reason="SessionManager not initialized. Please check API Keys configuration.")
        print("❌ WebSocket rejected: SessionManager not initialized")
        return
    
    await websocket.accept()
    print("✅ WebSocket connection accepted")

    session = await session_manager.create_session(websocket)
    print(f"📝 Session created: {session.id}")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                print("🔌 Client disconnected")
                break
            
            elif "text" in message:
                await session_manager.handle_text_message(session.id, message["text"])
            
            elif "bytes" in message:
                await session_manager.handle_binary_audio(session.id, message["bytes"])

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await session_manager.close_session(session.id)
        print(f"🔚 Session closed: {session.id}")

# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global run_logger, session_manager
    
    print("\n" + "=" * 60)
    print("🚀 Realtime Transcriber API Starting...")
    print("=" * 60)
    
    print(f"\n📊 Class availability check:")
    print(f"  - RunLogger class: {RunLogger is not None} ({RunLogger})")
    print(f"  - SessionManager class: {SessionManager is not None} ({SessionManager})")
    
    # 🔧 初始化 RunLogger
    if RunLogger is not None:
        try:
            run_logger = RunLogger(base_dir=Path("runs"))
            print(f"✅ RunLogger initialized: {run_logger}")
            print(f"✅ RunLogger base_dir: {run_logger.base_dir}")
        except Exception as e:
            print(f"❌ Failed to initialize RunLogger: {e}")
            import traceback
            traceback.print_exc()
            run_logger = None
    else:
        print("⚠️ RunLogger class not available (import failed)")
        run_logger = None
    
    # 🔧 ==================== Step 5: 创建 SessionManager ====================
    
    if SessionManager is not None:
        print("\n" + "="*60)
        print("🔧 STEP 5: Creating SessionManager")
        print("="*60)
        print(f"ELEVENLABS_API_KEY to pass: {'SET (' + str(len(ELEVENLABS_API_KEY)) + ' chars)' if ELEVENLABS_API_KEY else 'NOT SET'}")
        if ELEVENLABS_API_KEY:
            print(f"Preview: {ELEVENLABS_API_KEY[:10]}...{ELEVENLABS_API_KEY[-4:]}")
        print("="*60 + "\n")
        
        try:
            # 🔧 使用新的参数顺序：api_key 在前
            session_manager = SessionManager(
                api_key=ELEVENLABS_API_KEY,
                run_logger=run_logger
            )
            print(f"✅ SessionManager initialized: {session_manager}")
            
        except Exception as e:
            print(f"❌ Failed to initialize SessionManager: {e}")
            import traceback
            traceback.print_exc()
            session_manager = None
    else:
        print("⚠️ SessionManager class not available (import failed)")
        session_manager = None
    
    # 🔧 初始化 SummaryService（如果需要）
    try:
        from summary_service import SummaryService
        print(f"✅ SummaryService available")
    except Exception as e:
        print(f"⚠️ SummaryService not available: {e}")
    
    # 🔧 打印最终状态
    print(f"\n📊 Final initialization status:")
    print(f"  - run_logger: {run_logger is not None} ({run_logger})")
    print(f"  - session_manager: {session_manager is not None} ({session_manager})")
    
    if session_manager is None:
        print("\n🚨 CRITICAL: session_manager is None!")
        print("🚨 WebSocket connections will fail!")
        print("🚨 Please check the error messages above for the root cause.")
    
    # 🔇 设置日志模式
    log_mode = os.getenv("LOG_MODE", "quiet").lower()
    
    if log_mode == "verbose" or log_mode == "debug":
        LogConfig.enable_verbose()
    else:
        LogConfig.enable_quiet()
    
    LogConfig.print_config()
    
    # 🔧 验证和打印摘要配置
    try:
        SummaryConfig.validate()
        SummaryConfig.print_config()
    except Exception as e:
        print(f"⚠️  Summary config: {e}")
    
    # 🔧 打印所有路由（修复 Pylance 警告）
    print("\n📋 Registered Routes:")
    for route in app.routes:
        # 使用 getattr 安全访问属性
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", {"WS"})
        
        if path:
            print(f"  {str(methods):<20} {path}")
    
    print("\n" + "=" * 60)
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("📡 API Root:          http://127.0.0.1:8000/")
    print("📁 Projects API:      http://127.0.0.1:8000/api/projects")
    print("💚 Health Check:      http://127.0.0.1:8000/health")
    print("🔌 WebSocket:         ws://127.0.0.1:8000/ws/transcribe")
    print("=" * 60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    print("\n" + "=" * 60)
    print("🛑 Realtime Transcriber API Shutting Down...")
    print("=" * 60 + "\n")

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)