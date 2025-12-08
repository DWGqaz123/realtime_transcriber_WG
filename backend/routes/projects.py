# backend/routes/projects.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from database.db import DatabaseManager

router = APIRouter(prefix="/api/projects", tags=["projects"])

# ========== Pydantic Models ==========

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


class SessionResponse(BaseModel):
    id: int
    mode: str
    duration_seconds: int
    sentence_count: int
    char_count: int
    started_at: datetime
    ended_at: Optional[datetime]
    transcript_text: Optional[str] = ""
    
    class Config:
        from_attributes = True


# ========== Project Endpoints ==========

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """创建新项目"""
    print(f"\n{'='*60}")
    print(f"📝 Received create project request")
    print(f"   Name: '{project.name}'")
    print(f"   Description: '{project.description}'")
    print(f"{'='*60}\n")
    
    try:
        # 测试数据库连接
        print("🔍 Testing database connection...")
        if DatabaseManager is None:
            print("❌ DatabaseManager is None!")
            raise HTTPException(status_code=500, detail="Database not available")
        
        print("✅ DatabaseManager available")
        
        # 创建项目
        print(f"📝 Calling DatabaseManager.create_project...")
        db_project = DatabaseManager.create_project(
            name=project.name,
            description=project.description or ""
        )
        
        print(f"✅ Database returned project: {db_project}")
        print(f"   ID: {db_project.id}")
        print(f"   Name: {db_project.name}")
        print(f"   Sessions count: {len(db_project.sessions) if hasattr(db_project, 'sessions') else 'N/A'}")
        
        # 验证响应
        print(f"📦 Creating response model...")
        result = ProjectResponse(
            id=db_project.id,
            name=db_project.name,
            description=db_project.description,
            created_at=db_project.created_at,
            updated_at=db_project.updated_at,
            session_count=len(db_project.sessions))
        
        print(f"✅ Response created successfully")
        print(f"{'='*60}\n")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR in create_project:")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ProjectResponse])
async def get_all_projects():
    """获取所有项目"""
    try:
        projects = DatabaseManager.get_all_projects()
        return [
            ProjectResponse.model_validate({
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                "session_count": len(p.sessions)
            }, from_attributes=True)
            for p in projects
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    """获取单个项目"""
    try:
        project = DatabaseManager.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectResponse.model_validate({
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "session_count": len(project.sessions)
        }, from_attributes=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}")
async def delete_project(project_id: int):
    """删除项目"""
    try:
        success = DatabaseManager.delete_project(project_id)
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"message": "Project deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Session Endpoints ==========

@router.get("/{project_id}/sessions", response_model=List[SessionResponse])
async def get_project_sessions(project_id: int):
    """获取项目的所有会话（不含转录文本）"""
    try:
        sessions = DatabaseManager.get_project_sessions(project_id)
        return [
            SessionResponse(
                id=s.id,
                mode=s.mode,
                duration_seconds=s.duration_seconds,
                sentence_count=s.sentence_count,
                char_count=s.char_count,
                started_at=s.started_at,
                ended_at=s.ended_at,
                transcript_text=""  # 列表不返回完整转录
            )
            for s in sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/sessions/{session_id}", response_model=SessionResponse)
async def get_session_detail(project_id: int, session_id: int):
    """获取单个 session 的详细信息（含完整转录）"""
    try:
        session = DatabaseManager.get_session_by_id(session_id)
        
        if not session:
            print(f"❌ Session {session_id} not found in database")
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.project_id != project_id:
            print(f"❌ Session {session_id} belongs to project {session.project_id}, not {project_id}")
            raise HTTPException(status_code=404, detail="Session not found in this project")
        
        print(f"✅ Found session {session_id}: {len(session.transcript_text or '')} chars")
        
        return SessionResponse(
            id=session.id,
            mode=session.mode,
            duration_seconds=session.duration_seconds,
            sentence_count=session.sentence_count,
            char_count=session.char_count,
            started_at=session.started_at,
            ended_at=session.ended_at,
            transcript_text=session.transcript_text or ""
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting session detail: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ========== File Export Endpoint ==========

@router.get("/{project_id}/sessions/{session_id}/export")
async def export_session_file(project_id: int, session_id: int):
    """导出会话文件"""
    try:
        # 获取会话
        session = DatabaseManager.get_session_by_id(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 验证属于该项目
        if session.project_id != project_id:
            raise HTTPException(status_code=404, detail="Session not found in this project")
        
        # 获取项目
        project = DatabaseManager.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # 构建文件路径
        import os
        base_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "transcripts"

        
        safe_project_name = "".join(
            c for c in project.name 
            if c.isalnum() or c in (' ', '-', '_')
        ).strip()
        
        timestamp = session.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{session.mode}.txt"
        filepath = base_dir / safe_project_name / filename
        
        # 检查文件是否存在
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            raise HTTPException(status_code=404, detail="File not found")
        
        print(f"✅ Exporting file: {filepath}")
        
        # 返回文件
        return FileResponse(
            path=str(filepath),
            media_type="text/plain",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error exporting file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
