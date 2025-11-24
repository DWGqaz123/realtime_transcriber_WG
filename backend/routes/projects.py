# backend/routes/projects.py

# backend/routes/projects.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from database.db import DatabaseManager

# 🔧 确保这行正确
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
    
    class Config:
        from_attributes = True


# ========== Endpoints ==========

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """创建新项目"""
    try:
        db_project = DatabaseManager.create_project(
            name=project.name,
            description=project.description or ""
        )
        
        return ProjectResponse.model_validate(db_project, from_attributes=True)
        
    except Exception as e:
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


@router.get("/{project_id}/sessions", response_model=List[SessionResponse])
async def get_project_sessions(project_id: int):
    """获取项目的所有会话"""
    try:
        sessions = DatabaseManager.get_project_sessions(project_id)
        return [
            SessionResponse.model_validate(s, from_attributes=True)
            for s in sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))