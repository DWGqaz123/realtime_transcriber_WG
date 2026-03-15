# backend/main.py

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import sys
import logging

log = logging.getLogger("transcriber.main")


def get_backend_host() -> str:
    return os.getenv("HOST", "0.0.0.0")


def get_backend_port() -> int:
    try:
        return int(os.getenv("PORT", "9123"))
    except ValueError:
        return 9123

if getattr(sys, 'frozen', False):
    # 打包环境
    log_file = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "backend.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 重定向 stdout 和 stderr
    sys.stdout = open(log_file, 'w', buffering=1)
    sys.stderr = sys.stdout
    
    log.info("Backend log started: %s", log_file)

# ==================== 加载环境变量 ====================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# ==================== 导入配置和服务 ====================

from config import LogConfig, SummaryConfig, ELEVENLABS_API_KEY
from session_manager import SessionManager
from routes import projects, search

# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期事件"""
    global run_logger, session_manager
    
    log.info("Realtime Transcriber API starting...")
    
    try:
        session_manager = SessionManager(api_key=ELEVENLABS_API_KEY)
        log.info("SessionManager initialized")
    except Exception as e:
        log.error("Failed to initialize SessionManager: %s", e, exc_info=True)
        session_manager = None
    
    if session_manager is None:
        log.critical("session_manager is None — WebSocket connections will fail")
    
    # 设置日志模式
    log_mode = os.getenv("LOG_MODE", "quiet").lower()
    if log_mode in ("verbose", "debug"):
        LogConfig.enable_verbose()
    else:
        LogConfig.enable_quiet()
    
    LogConfig.print_config()
    
    SummaryConfig.print_config()
    
    public_host = os.getenv("PUBLIC_HOST", os.getenv("HOST", "127.0.0.1"))
    public_port = get_backend_port()
    log.info(
        "API ready — docs: http://%s:%d/docs, ws: ws://%s:%d/ws/transcribe",
        public_host,
        public_port,
        public_host,
        public_port,
    )
    
    yield
    
    log.info("Realtime Transcriber API shutting down")

# ==================== 创建 FastAPI 应用 ====================

app = FastAPI(
    title="Realtime Transcriber API",
    description="API for real-time transcription with project management",
    version="2.0.0",
    lifespan=lifespan,
)

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
app.include_router(search.router)

# ==================== 全局变量 ====================

session_manager: Optional[SessionManager] = None
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
        "session_manager_ready": session_manager is not None
    }


@app.get("/api/config/summary")
async def get_summary_config():
    """Expose summary timing config for frontend synchronization."""
    return {
        "summary_interval_seconds": SummaryConfig.SUMMARY_INTERVAL_SECONDS,
        "loose_mode_threshold": SummaryConfig.LOOSE_MODE_THRESHOLD,
        "min_sentences": SummaryConfig.MIN_SENTENCES,
    }

# ==================== WebSocket 端点 ====================

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """实时转录 WebSocket 端点"""
    
    if session_manager is None:
        await websocket.close(code=1011, reason="SessionManager not initialized.")
        log.warning("WebSocket rejected: SessionManager not initialized")
        return
    
    await websocket.accept()
    session = await session_manager.create_session(websocket)
    log.info("WebSocket session created: %s", session.id)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                log.info("Client disconnected")
                break
            
            elif "text" in message:
                await session_manager.handle_text_message(session.id, message["text"])
            
            elif "bytes" in message:
                await session_manager.handle_binary_audio(session.id, message["bytes"])

    except Exception as e:
        log.error("WebSocket error: %s", e, exc_info=True)

    finally:
        await session_manager.close_session(session.id)
        log.info("Session closed: %s", session.id)

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=get_backend_host(), port=get_backend_port())
