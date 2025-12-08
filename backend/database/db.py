# backend/database/db.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DBSession, joinedload
from database.models import Base, Project, Session
from typing import Optional, List
from datetime import datetime
import os
from pathlib import Path
from database.models import Project, Session, Summary  

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
    @staticmethod
    def _init_db():
        """初始化数据库"""
        if DatabaseManager._engine is None:
            db_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber"
            db_dir.mkdir(parents=True, exist_ok=True)
            
            db_path = db_dir / "transcripts.db"
            database_url = f"sqlite:///{db_path}"
            
            DatabaseManager._engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False}
            )
            
            DatabaseManager._SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=DatabaseManager._engine
            )
            
            # 🔧 创建所有表
            from database.models import Base
            Base.metadata.create_all(bind=DatabaseManager._engine)
            
            print(f"✅ Database initialized at: {db_path}")
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
    
    # ========== Summary 操作 ==========

    @staticmethod
    def create_summary(
        session_id: int,
        project_id: int,
        content: str,
        source_text: str,
        start_sentence_idx: int,
        end_sentence_idx: int,
        duration_seconds: int = 0
    ) -> Summary:
        """创建摘要"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            from datetime import datetime
            
            summary = Summary(
                session_id=session_id,
                project_id=project_id,
                content=content,
                source_text=source_text,
                start_sentence_idx=start_sentence_idx,
                end_sentence_idx=end_sentence_idx,
                duration_seconds=duration_seconds,
                created_at=datetime.utcnow(),
                embedding_status="pending"
            )
            
            db.add(summary)
            db.commit()
            db.refresh(summary)
            
            print(f"✅ [DB] Created summary: id={summary.id}, session={session_id}")
            
            return summary
            
        except Exception as e:
            db.rollback()
            print(f"❌ [DB] Error creating summary: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            db.close()


    @staticmethod
    def get_session_summaries(session_id: int) -> List[Summary]:
        """获取会话的所有摘要"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            
            summaries = db.query(Summary).filter(
                Summary.session_id == session_id
            ).order_by(Summary.created_at.asc()).all()
            
            return summaries
            
        finally:
            db.close()


    @staticmethod
    def get_project_summaries(project_id: int) -> List[Summary]:
        """获取项目的所有摘要"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            
            summaries = db.query(Summary).filter(
                Summary.project_id == project_id
            ).order_by(Summary.created_at.desc()).all()
            
            return summaries
            
        finally:
            db.close()


    @staticmethod
    def get_summary_by_id(summary_id: int) -> Optional[Summary]:
        """根据 ID 获取摘要"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            
            summary = db.query(Summary).filter(Summary.id == summary_id).first()
            return summary
            
        finally:
            db.close()


    @staticmethod
    def update_summary_embedding(
        summary_id: int,
        embedding_vector: str,
        status: str = "completed"
    ) -> bool:
        """更新摘要的 embedding 信息（预留）"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            
            summary = db.query(Summary).filter(Summary.id == summary_id).first()
            
            if not summary:
                return False
            
            summary.embedding_vector = embedding_vector
            summary.embedding_status = status
            
            db.commit()
            
            print(f"✅ [DB] Updated embedding for summary {summary_id}")
            
            return True
            
        except Exception as e:
            db.rollback()
            print(f"❌ [DB] Error updating embedding: {e}")
            raise
        finally:
            db.close()


    @staticmethod
    def get_pending_embeddings(limit: int = 100) -> List[Summary]:
        """获取待向量化的摘要（预留）"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            
            summaries = db.query(Summary).filter(
                Summary.embedding_status == "pending"
            ).limit(limit).all()
            
            return summaries
            
        finally:
            db.close()