# backend/main.py

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
# ======= Import DatabaseManager =======
try:
    from database.db import DatabaseManager
    print("✅ DatabaseManager imported successfully")
except Exception as e:
    print(f"❌ Failed to import DatabaseManager: {e}")
    DatabaseManager = None


# ======= DON'T INITIALIZE SessionManager HERE =======
# ⭐️ FIXED: 不能在 import 阶段初始化 session_manager !!!
session_manager = None
run_logger = None


# ======= Import class definitions only =======
from pathlib import Path  # 👈 新增

# 导入会话管理器和日志（只导入类，不在这里实例化）
try:
    from session_manager import SessionManager
    from run_logger import RunLogger
    print("✅ SessionManager and RunLogger imported successfully")
except Exception as e:
    print(f"⚠️ Warning: SessionManager not available: {e}")
    SessionManager = None
    RunLogger = None

# 👇 全局实例变量，初始为 None，由 startup_event 里创建
run_logger: Optional["RunLogger"] = None
session_manager: Optional["SessionManager"] = None

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


# ========== Pydantic 模型 ==========

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    session_count: int
    
    class Config:
        from_attributes = True


# ========== 项目 API 端点（不改动） ==========

@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_all_projects():
    if DatabaseManager is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        projects = DatabaseManager.get_all_projects()
        print(f"📁 Retrieved {len(projects)} projects")
        
        return [
            ProjectResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                created_at=p.created_at,
                updated_at=p.updated_at,
                session_count=len(p.sessions)
            )
            for p in projects
        ]
    except Exception as e:
        print(f"❌ Error in get_all_projects: {e}")
        raise


@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    if DatabaseManager is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    try:
        db_project = DatabaseManager.create_project(
            name=project.name,
            description=project.description or ""
        )
        print(f"✅ Created project: {db_project.name}")
        
        return ProjectResponse(
            id=db_project.id,
            name=db_project.name,
            description=db_project.description,
            created_at=db_project.created_at,
            updated_at=db_project.updated_at,
            session_count=0
        )
    except Exception as e:
        print(f"❌ Error in create_project: {e}")
        raise


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    if DatabaseManager is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    project = DatabaseManager.get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        updated_at=project.updated_at,
        session_count=len(project.sessions)
    )


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: int):
    if DatabaseManager is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    success = DatabaseManager.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    
    print(f"🗑️ Deleted project: {project_id}")
    return {"message": "Project deleted successfully"}


@app.get("/api/projects/{project_id}/sessions")
async def get_project_sessions(project_id: int):
    if DatabaseManager is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    sessions = DatabaseManager.get_project_sessions(project_id)
    return [
        {
            "id": s.id,
            "mode": s.mode,
            "duration_seconds": s.duration_seconds,
            "sentence_count": s.sentence_count,
            "char_count": s.char_count,
            "started_at": s.started_at.isoformat(),
            "ended_at": s.ended_at.isoformat() if s.ended_at else None
        }
        for s in sessions
    ]


# ========== 基础端点 ==========

@app.get("/")
async def root():
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
    return {
        "status": "healthy",
        "database": "available" if DatabaseManager else "unavailable",
        "session_manager": "available" if session_manager else "unavailable"
    }


# ========= ⭐️ FIXED: WebSocket 端点 =========

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()
    print("✅ WebSocket connection accepted")
    print(f"[{datetime.now()}] session_manager is:", session_manager)

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

    finally:
        await session_manager.close_session(session.id)
        print(f"🔚 Session closed: {session.id}")


# ========= ⭐️ FIXED: Startup: 正确初始化 session_manager =========

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    global run_logger, session_manager  # 👈 一定要声明 global 才能修改上面的全局变量

    print("\n" + "=" * 60)
    print("🚀 Realtime Transcriber API Starting...")
    print("=" * 60)

    # ⭐️ 初始化 RunLogger
    if RunLogger is not None:
        try:
            # 关键修复点：传 Path 对象，而不是 str，这样 base_dir.mkdir() 就合法了
            run_logger = RunLogger(base_dir=Path("runs"))
            print(f"✅ RunLogger initialized with base_dir={run_logger.base_dir}")
        except Exception as e:
            print(f"❌ Failed to initialize RunLogger: {e}")
            run_logger = None
    else:
        print("⚠️ RunLogger class not available")

    # ⭐️ 初始化 SessionManager
    if SessionManager is not None:
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            if not api_key:
                print("⚠️ ELEVENLABS_API_KEY is empty or not set (SessionManager will still be created, but ElevenLabs will fail until key is set).")
            session_manager = SessionManager(
                run_logger=run_logger,
                api_key=api_key,
            )
            print("✅ SessionManager initialized:", session_manager)
        except Exception as e:
            print(f"❌ Failed to initialize SessionManager: {e}")
            session_manager = None
    else:
        print("⚠️ SessionManager class not available")

    # 下面保留你原来打印路由的代码（可选）
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

# ========= Shutdown =========

@app.on_event("shutdown")
async def shutdown_event():
    print("\n========== 🛑 Shutdown ==========\n")


# ========= Main =========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)