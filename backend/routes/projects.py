# backend/routes/projects.py

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database.db import DatabaseManager

log = logging.getLogger("transcriber.routes.projects")

router = APIRouter(prefix="/api/projects", tags=["projects"])


# ── Guard helpers ─────────────────────────────────────────────────────────────

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


# ── Path / response builders ──────────────────────────────────────────────────

def build_transcript_filepath(project_name: str, session) -> Path:
    safe_name = "".join(c for c in project_name if c.isalnum() or c in (" ", "-", "_")).strip()
    timestamp = session.started_at.strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path.home()
        / "Library" / "Application Support" / "RealtimeTranscriber"
        / "transcripts" / safe_name
        / f"{timestamp}_{session.mode}.txt"
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


def build_session_response(session, summaries=None, transcript_text=None) -> "SessionResponse":
    return SessionResponse(
        id=session.id,
        mode=session.mode,
        name=session.name,
        notes=session.notes,
        duration_seconds=session.duration_seconds,
        sentence_count=session.sentence_count,
        char_count=session.char_count,
        started_at=session.started_at,
        ended_at=session.ended_at,
        transcript_text=transcript_text if transcript_text is not None else (session.transcript_text or ""),
        summaries=summaries,
    )


# ── Pydantic models ───────────────────────────────────────────────────────────

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


class SessionUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None


class SessionResponse(BaseModel):
    id: int
    mode: str
    name: Optional[str] = None
    notes: Optional[str] = None
    duration_seconds: int
    sentence_count: int
    char_count: int
    started_at: datetime
    ended_at: Optional[datetime]
    transcript_text: Optional[str] = ""
    summaries: Optional[List["SummaryResponse"]] = []

    class Config:
        from_attributes = True


# ── Project endpoints ─────────────────────────────────────────────────────────

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    db_project = DatabaseManager.create_project(
        name=project.name,
        description=project.description or "",
    )
    return build_project_response(db_project)


@router.get("/", response_model=List[ProjectResponse])
async def get_all_projects():
    return [build_project_response(p) for p in DatabaseManager.get_all_projects()]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    return build_project_response(require_project(project_id))


@router.delete("/{project_id}")
async def delete_project(project_id: int):
    if not DatabaseManager.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully"}


# ── Session endpoints ─────────────────────────────────────────────────────────

@router.get("/{project_id}/sessions", response_model=List[SessionResponse])
async def get_project_sessions(project_id: int):
    sessions = DatabaseManager.get_project_sessions(project_id)
    return [build_session_response(s, transcript_text="") for s in sessions]


@router.patch("/{project_id}/sessions/{session_id}", response_model=SessionResponse)
async def update_session_metadata(project_id: int, session_id: int, body: SessionUpdate):
    session = require_project_session(project_id, session_id)
    updated = DatabaseManager.update_session(
        session_id=session.id,
        name=body.name,
        notes=body.notes,
    )
    return build_session_response(updated)


@router.get("/{project_id}/sessions/{session_id}", response_model=SessionResponse)
async def get_session_detail(project_id: int, session_id: int):
    session = require_project_session(project_id, session_id)
    summaries = DatabaseManager.get_session_summaries(session_id)
    summary_responses = [
        SummaryResponse(
            id=s.id,
            content=s.content,
            created_at=s.created_at,
            sentence_count=max(0, (s.end_sentence_idx or 0) - (s.start_sentence_idx or 0)),
            duration_seconds=s.duration_seconds,
            start_sentence_idx=s.start_sentence_idx,
            end_sentence_idx=s.end_sentence_idx,
        )
        for s in summaries
    ]
    return build_session_response(session, summaries=summary_responses)


@router.get("/{project_id}/sessions/{session_id}/export")
async def export_session_file(project_id: int, session_id: int):
    session = require_project_session(project_id, session_id)
    project = require_project(project_id)
    filepath = build_transcript_filepath(project.name, session)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(filepath), media_type="text/plain", filename=filepath.name)


@router.delete("/{project_id}/sessions/{session_id}")
async def delete_session(project_id: int, session_id: int):
    session = require_project_session(project_id, session_id)
    project = require_project(project_id)

    try:
        filepath = build_transcript_filepath(project.name, session)
        if filepath.exists():
            os.remove(filepath)
    except Exception as e:
        log.warning("Failed to delete transcript file: %s", e)

    if not DatabaseManager.delete_session(session_id):
        raise HTTPException(status_code=500, detail="Failed to delete session")
    return {"message": "Session deleted successfully", "session_id": session_id}


@router.delete("/{project_id}/sessions/{session_id}/summaries/{summary_id}")
async def delete_summary(project_id: int, session_id: int, summary_id: int):
    require_project_summary(project_id, session_id, summary_id)
    if not DatabaseManager.delete_summary(summary_id):
        raise HTTPException(status_code=500, detail="Failed to delete summary")
    return {"message": "Summary deleted successfully", "summary_id": summary_id}
