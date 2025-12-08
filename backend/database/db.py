# backend/database/db.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DBSession, joinedload
from database.models import Base, Project, Session
from typing import Optional, List
from datetime import datetime
import os

# 数据库文件路径
DB_PATH = os.path.expanduser("~/Library/Application Support/RealtimeTranscriber/transcripts.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 创建数据库引擎
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

# 创建表
Base.metadata.create_all(engine)

# 会话工厂
SessionLocal = sessionmaker(bind=engine)


class DatabaseManager:
    """数据库管理器"""
    
    @staticmethod
    def get_db() -> DBSession:
        """获取数据库会话"""
        db = SessionLocal()
        try:
            return db
        except:
            db.close()
            raise
    
    # ========== Project 操作 ==========
    
    @staticmethod
    def create_project(name: str, description: str = "") -> Project:
        """创建新项目"""
        db = DatabaseManager.get_db()
        try:
            project = Project(name=name, description=description)
            db.add(project)
            db.commit()
            db.refresh(project)
            
            # 🔧 初始化 sessions 列表（空列表）
            project.sessions = []
            
            return project
        finally:
            db.close()
    
    @staticmethod
    def get_all_projects() -> List[Project]:
        """获取所有项目"""
        db = DatabaseManager.get_db()
        try:
            # 🔧 使用 joinedload 预加载 sessions 关系
            projects = db.query(Project).options(
                joinedload(Project.sessions)
            ).order_by(Project.updated_at.desc()).all()
            
            # 🔧 确保 sessions 已加载（触发懒加载）
            for project in projects:
                _ = len(project.sessions)
            
            return projects
        finally:
            db.close()
    
    @staticmethod
    def get_project_by_id(project_id: int) -> Optional[Project]:
        """根据 ID 获取项目"""
        db = DatabaseManager.get_db()
        try:
            project = db.query(Project).options(
                joinedload(Project.sessions)
            ).filter(Project.id == project_id).first()
            
            if project:
                _ = len(project.sessions)
            
            return project
        finally:
            db.close()
    
    @staticmethod
    def delete_project(project_id: int) -> bool:
        """删除项目（会级联删除所有会话）"""
        db = DatabaseManager.get_db()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                db.delete(project)
                db.commit()
                return True
            return False
        finally:
            db.close()
    
    # ========== Session 操作 ==========
    
    @staticmethod
    def create_session(project_id: int, mode: str) -> Session:
        """创建新会话"""
        db = DatabaseManager.get_db()
        try:
            session = Session(project_id=project_id, mode=mode)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session
        finally:
            db.close()
    
    @staticmethod
    def update_session(
        session_id: int,
        duration_seconds: Optional[int] = None,
        transcript_text: Optional[str] = None,
        sentence_count: Optional[int] = None,
        char_count: Optional[int] = None,
        ended_at: Optional[datetime] = None
    ) -> Optional[Session]:
        """更新会话信息"""
        db = DatabaseManager.get_db()
        try:
            session = db.query(Session).filter(Session.id == session_id).first()
            if session:
                # 构建更新字典
                updates = {}
                if duration_seconds is not None:
                    updates["duration_seconds"] = duration_seconds
                if transcript_text is not None:
                    updates["transcript_text"] = transcript_text
                if sentence_count is not None:
                    updates["sentence_count"] = sentence_count
                if char_count is not None:
                    updates["char_count"] = char_count
                if ended_at is not None:
                    updates["ended_at"] = ended_at
                
                # 批量更新
                for key, value in updates.items():
                    setattr(session, key, value)
                
                db.commit()
                db.refresh(session)
                return session
            return None
        finally:
            db.close()
    
    @staticmethod
    def get_project_sessions(project_id: int) -> List[Session]:
        """获取项目的所有会话"""
        db = DatabaseManager.get_db()
        try:
            sessions = db.query(Session).filter(
                Session.project_id == project_id
            ).order_by(Session.started_at.desc()).all()
            
            return sessions
        finally:
            db.close()
    
    @staticmethod
    def get_session_by_id(session_id: int) -> Optional[Session]:
        """根据 ID 获取会话"""
        db = DatabaseManager.get_db()
        try:
            session = db.query(Session).filter(Session.id == session_id).first()
            return session
        finally:
            db.close()
    
    