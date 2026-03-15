# backend/routes/projects.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import logging

from database.db import DatabaseManager

log = logging.getLogger("transcriber.routes.projects")

router = APIRouter(prefix="/api/projects", tags=["projects"])


def require_project(project_id: int):
    project = DatabaseManager.get_project_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def require_project_session(project_id: int, session_id: int):
    session = DatabaseManager.get_session_by_id(session_id)
    if session is None or session.project_id != project_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def require_project_summary(project_id: int, session_id: int, summary_id: int):
    summary = DatabaseManager.get_summary_by_id(summary_id)
    if summary is None or summary.session_id != session_id:
        raise HTTPException(status_code=404, detail="Summary not found")
    session = DatabaseManager.get_session_by_id(session_id)
    if session is None or session.project_id != project_id:
        raise HTTPException(status_code=404, detail="Summary not found")
    return summary


def build_transcript_filepath(project_name: str, session) -> Path:
    safe_project_name = "".join(
        c for c in project_name
        if c.isalnum() or c in (" ", "-", "_")
    ).strip()
    timestamp = session.started_at.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{session.mode}.txt"
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "RealtimeTranscriber"
        / "transcripts"
        / safe_project_name
        / filename
    )


def build_project_response(project) -> "ProjectResponse":
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        updated_at=project.updated_at,
        session_count=getattr(project, "session_count", 0),
    )


def build_session_response(
    session,
    summaries: Optional[List["SummaryResponse"]] = None,
    transcript_text: Optional[str] = None
) -> "SessionResponse":
    return SessionResponse(
        id=session.id,
        mode=session.mode,
        duration_seconds=session.duration_seconds,
        sentence_count=session.sentence_count,
        char_count=session.char_count,
        started_at=session.started_at,
        ended_at=session.ended_at,
        transcript_text=transcript_text if transcript_text is not None else (session.transcript_text or ""),
        summaries=summaries,
    )

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

class SummaryResponse(BaseModel):
    id: int
    content: str
    created_at: datetime
    sentence_count: int
    duration_seconds: int
    start_sentence_idx: int
    end_sentence_idx: int
    
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
    summaries: Optional[List["SummaryResponse"]] = []
    
    class Config:
        from_attributes = True


# ========== Project Endpoints ==========

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """创建新项目"""
    
    try:
        # 创建项目
        db_project = DatabaseManager.create_project(
            name=project.name,
            description=project.description or ""
        )
        
        
        # 验证响应
        return build_project_response(db_project)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ProjectResponse])
async def get_all_projects():
    """获取所有项目"""
    try:
        projects = DatabaseManager.get_all_projects()
        return [build_project_response(project) for project in projects]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    """获取单个项目"""
    try:
        return build_project_response(require_project(project_id))
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
            build_session_response(s, transcript_text="")
            for s in sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}/sessions/{session_id}", response_model=SessionResponse)
async def get_session_detail(project_id: int, session_id: int):
    """获取单个 session 的详细信息（含完整转录）"""
    try:
        session = require_project_session(project_id, session_id)
        summaries = DatabaseManager.get_session_summaries(session_id)
        
        summary_responses = [
            SummaryResponse(
                id=s.id,
                content=s.content,
                created_at=s.created_at,
                sentence_count=s.end_sentence_idx - s.start_sentence_idx + 1,
                duration_seconds=s.duration_seconds,
                start_sentence_idx=s.start_sentence_idx,
                end_sentence_idx=s.end_sentence_idx
            )
            for s in summaries
        ]
        
        return build_session_response(session, summaries=summary_responses)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== File Export Endpoint ==========

@router.get("/{project_id}/sessions/{session_id}/export")
async def export_session_file(project_id: int, session_id: int):
    """导出会话文件"""
    try:
        session = require_project_session(project_id, session_id)
        project = require_project(project_id)
        filepath = build_transcript_filepath(project.name, session)
        
        # 检查文件是否存在
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        
        # 返回文件
        return FileResponse(
            path=str(filepath),
            media_type="text/plain",
            filename=filepath.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{project_id}/sessions/{session_id}")
async def delete_session(project_id: int, session_id: int):
    """删除 session（级联删除 summaries 和转录文件）"""
    try:
        import os
        session = require_project_session(project_id, session_id)
        project = require_project(project_id)
        
        # 删除转录文件（如果存在）
        try:
            filepath = build_transcript_filepath(project.name, session)
            if filepath.exists():
                os.remove(filepath)
                
        except FileNotFoundError:
            pass  # 文件已不存在，无需处理
        except Exception as e:
            log.warning("Failed to delete transcript file %s: %s", filepath, e)
        
        success = DatabaseManager.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete session")
        
        
        return {
            "message": "Session deleted successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}/sessions/{session_id}/summaries/{summary_id}")
async def delete_summary(project_id: int, session_id: int, summary_id: int):
    """删除单个摘要"""
    try:
        require_project_summary(project_id, session_id, summary_id)
        
        success = DatabaseManager.delete_summary(summary_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete summary")
        
        
        return {
            "message": "Summary deleted successfully",
            "summary_id": summary_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
