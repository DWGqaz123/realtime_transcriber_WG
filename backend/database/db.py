# backend/database/db.py

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session as DBSession
from database.models import Base, Project, Session, Summary
from typing import Optional, List
from datetime import datetime
from pathlib import Path


class DatabaseManager:
    """数据库管理器"""
    
    _engine = None
    _SessionLocal = None
    
    @staticmethod
    def _table_has_column(connection, table_name: str, column_name: str) -> bool:
        rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
        return any(row[1] == column_name for row in rows)

    @staticmethod
    def _migrate_remove_redundant_project_columns():
        engine = DatabaseManager._engine
        if engine is None:
            return

        with engine.begin() as connection:
            needs_summary_migration = DatabaseManager._table_has_column(connection, "summaries", "project_id")
            needs_embedding_migration = DatabaseManager._table_has_column(connection, "embeddings", "project_id")

            if not needs_summary_migration and not needs_embedding_migration:
                return

            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")

            if needs_summary_migration:
                connection.exec_driver_sql("ALTER TABLE summaries RENAME TO summaries_old")
                connection.exec_driver_sql(
                    """
                    CREATE TABLE summaries (
                        id INTEGER PRIMARY KEY,
                        session_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        source_text TEXT,
                        start_sentence_idx INTEGER,
                        end_sentence_idx INTEGER,
                        created_at DATETIME,
                        duration_seconds INTEGER,
                        is_indexed BOOLEAN,
                        indexed_at DATETIME,
                        FOREIGN KEY(session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                    """
                )
                connection.exec_driver_sql(
                    """
                    INSERT INTO summaries (
                        id, session_id, content, source_text, start_sentence_idx,
                        end_sentence_idx, created_at, duration_seconds, is_indexed, indexed_at
                    )
                    SELECT
                        id, session_id, content, source_text, start_sentence_idx,
                        end_sentence_idx, created_at, duration_seconds, is_indexed, indexed_at
                    FROM summaries_old
                    """
                )
                connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_summaries_id ON summaries (id)")
                connection.exec_driver_sql("DROP TABLE summaries_old")

            if needs_embedding_migration:
                connection.exec_driver_sql("ALTER TABLE embeddings RENAME TO embeddings_old")
                connection.exec_driver_sql(
                    """
                    CREATE TABLE embeddings (
                        id INTEGER PRIMARY KEY,
                        summary_id INTEGER NOT NULL UNIQUE,
                        faiss_index_id INTEGER NOT NULL,
                        content_preview TEXT,
                        session_mode VARCHAR(50),
                        indexed_at DATETIME,
                        embedding_model VARCHAR(100),
                        embedding_dimension INTEGER,
                        FOREIGN KEY(summary_id) REFERENCES summaries (id) ON DELETE CASCADE
                    )
                    """
                )
                connection.exec_driver_sql(
                    """
                    INSERT INTO embeddings (
                        id, summary_id, faiss_index_id, content_preview,
                        session_mode, indexed_at, embedding_model, embedding_dimension
                    )
                    SELECT
                        id, summary_id, faiss_index_id, content_preview,
                        session_mode, indexed_at, embedding_model, embedding_dimension
                    FROM embeddings_old
                    """
                )
                connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_embeddings_id ON embeddings (id)")
                connection.exec_driver_sql("DROP TABLE embeddings_old")

            connection.exec_driver_sql("PRAGMA foreign_keys=ON")

    @staticmethod
    def get_db() -> DBSession:
        """获取数据库会话（裸返回，调用方负责关闭）"""
        if DatabaseManager._SessionLocal is None:
            DatabaseManager._init_db()
        return DatabaseManager._SessionLocal()

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
            
            Base.metadata.create_all(bind=DatabaseManager._engine)
            DatabaseManager._migrate_remove_redundant_project_columns()
            
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
            rows = db.query(
                Project,
                func.count(Session.id).label("session_count")
            ).outerjoin(
                Session, Session.project_id == Project.id
            ).group_by(
                Project.id
            ).order_by(
                Project.updated_at.desc()
            ).all()

            projects = []
            for project, session_count in rows:
                project.session_count = session_count
                projects.append(project)

            return projects
        finally:
            db.close()
    
    @staticmethod
    def get_project_by_id(project_id: int) -> Optional[Project]:
        """根据 ID 获取项目"""
        db = DatabaseManager.get_db()
        try:
            row = db.query(
                Project,
                func.count(Session.id).label("session_count")
            ).outerjoin(
                Session, Session.project_id == Project.id
            ).filter(
                Project.id == project_id
            ).group_by(
                Project.id
            ).first()

            if row is None:
                return None

            project, session_count = row
            project.session_count = session_count
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
                content=content,
                source_text=source_text,
                start_sentence_idx=start_sentence_idx,
                end_sentence_idx=end_sentence_idx,
                duration_seconds=duration_seconds,
                created_at=datetime.utcnow()            )
            
            db.add(summary)
            db.commit()
            db.refresh(summary)
            
            
            return summary
            
        except Exception:
            db.rollback()
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
            
            summaries = db.query(Summary).join(Session).filter(
                Session.project_id == project_id
            ).order_by(Summary.created_at.desc()).all()
            
            return summaries
            
        finally:
            db.close()


    @staticmethod
    def get_summary_by_id(summary_id: int):
        """通过 ID 获取摘要"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            return db.query(Summary).filter(Summary.id == summary_id).first()
        finally:
            db.close()


    @staticmethod
    def delete_session(session_id: int) -> bool:
        """
        删除 session（级联删除 summaries）
        
        Note: summaries 会通过数据库外键的 CASCADE 自动删除
        """
        db = DatabaseManager.get_db()
        try:
            from database.models import Session
            
            session = db.query(Session).filter(Session.id == session_id).first()
            
            if not session:
                return False

            db.delete(session)
            db.commit()
            
            
            return True
        
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
            
    
    @staticmethod
    def delete_summary(summary_id: int) -> bool:
        """删除摘要"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Summary
            
            summary = db.query(Summary).filter(Summary.id == summary_id).first()
            
            if not summary:
                return False
            
            db.delete(summary)
            db.commit()
            
            
            return True
            
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
