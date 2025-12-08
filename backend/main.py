# backend/main.py

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

# ======= Import DatabaseManager =======
try:
    from database.db import DatabaseManager
    print("✅ DatabaseManager imported successfully")
except Exception as e:
    print(f"❌ Failed to import DatabaseManager: {e}")
    import traceback
    traceback.print_exc()
    DatabaseManager = None

# ======= Import SessionManager and RunLogger =======
SessionManager = None
RunLogger = None

try:
    from run_logger import RunLogger
    print("✅ RunLogger imported successfully")
except Exception as e:
    print(f"❌ Failed to import RunLogger: {e}")
    import traceback
    traceback.print_exc()
    RunLogger = None

try:
    from session_manager import SessionManager
    print("✅ SessionManager imported successfully")
except Exception as e:
    print(f"❌ Failed to import SessionManager: {e}")
    import traceback
    traceback.print_exc()
    SessionManager = None

# ======= Import Routes =======
try:
    from routes import projects
    print("✅ Projects router imported successfully")
except Exception as e:
    print(f"❌ Failed to import projects router: {e}")
    import traceback
    traceback.print_exc()

# 🔧 全局变量
run_logger = None
session_manager = None

print(f"📊 After imports: SessionManager={SessionManager}, RunLogger={RunLogger}")

# ========== 创建 FastAPI 应用 ==========

app = FastAPI(
    title="Realtime Transcriber API",
    description="API for real-time transcription with project management",
    version="2.0.0"
)

print("✅ FastAPI app created")

# ========== CORS 配置 ==========

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 注册路由 ==========

app.include_router(projects.router)
print("✅ Projects router registered")

# ========== 基础端点 ==========

@app.get("/")
async def root():
    """API 根端点"""
    return {
        "message": "Realtime Transcriber API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "projects": "/api/projects",
            "websocket": "/ws/transcribe"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "database": "available" if DatabaseManager else "unavailable",
        "session_manager": "available" if session_manager else "unavailable"
    }


# ========== WebSocket 端点 ==========

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """实时转录 WebSocket 端点"""
    await websocket.accept()
    print("✅ WebSocket connection accepted")
    print(f"[{datetime.now()}] session_manager is: {session_manager}")

    if session_manager is None:
        await websocket.send_text("ERROR: session_manager not available on server")
        await websocket.close(code=1011, reason="Session manager not available")
        return

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


# ========== 启动事件 ==========

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
    
    # 初始化 RunLogger
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
    
    # 初始化 SessionManager
    if SessionManager is not None:
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            print(f"📌 ELEVENLABS_API_KEY: {'[SET]' if api_key else '[NOT SET]'}")
            
            if not api_key:
                print("⚠️ ELEVENLABS_API_KEY is empty")
            
            print(f"📌 Creating SessionManager with run_logger={run_logger}")
            session_manager = SessionManager(
                run_logger=run_logger,
                api_key=api_key,
            )
            print(f"✅ SessionManager initialized: {session_manager}")
            print(f"✅ SessionManager type: {type(session_manager)}")
            print(f"✅ session_manager is not None: {session_manager is not None}")
            
        except Exception as e:
            print(f"❌ Failed to initialize SessionManager: {e}")
            import traceback
            traceback.print_exc()
            session_manager = None
    else:
        print("⚠️ SessionManager class not available (import failed)")
        session_manager = None
    
    try:
        from summary_service import get_summary_service
        summary_service = get_summary_service()
        print(f"✅ SummaryService initialized (API key: {bool(summary_service.api_key)})")
    except Exception as e:
        print(f"⚠️ Failed to initialize SummaryService: {e}")
        
    # 打印最终状态
    print(f"\n📊 Final initialization status:")
    print(f"  - run_logger: {run_logger is not None} ({run_logger})")
    print(f"  - session_manager: {session_manager is not None} ({session_manager})")
    
    if session_manager is None:
        print("\n🚨 CRITICAL: session_manager is None!")
        print("🚨 WebSocket connections will fail!")
        print("🚨 Please check the error messages above for the root cause.")
    
    # 打印所有路由
    print("\n📋 Registered Routes:")
    for route in app.routes:
        if hasattr(route, "path"):
            methods = route.methods if hasattr(route, "methods") else {"WS"}
            print(f"  {str(methods):<20} {route.path}")
    
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


# ========== Main ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)