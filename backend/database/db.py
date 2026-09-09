# backend/database/db.py

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session as DBSession
from database.models import Base, Project, Session, Summary
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import logging

log = logging.getLogger("transcriber.db")


class DatabaseManager:

    _engine = None
    _SessionLocal = None

    @staticmethod
    def _table_has_column(connection, table_name: str, column_name: str) -> bool:
        rows = connection.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
        return any(row[1] == column_name for row in rows)

    @staticmethod
    def _migrate_add_session_name_notes():
        engine = DatabaseManager._engine
        if engine is None:
            return
        with engine.begin() as connection:
            existing = {
                row[1]
                for row in connection.exec_driver_sql("PRAGMA table_info(sessions)").fetchall()
            }
            if "name" not in existing:
                connection.exec_driver_sql("ALTER TABLE sessions ADD COLUMN name VARCHAR(255)")
            if "notes" not in existing:
                connection.exec_driver_sql("ALTER TABLE sessions ADD COLUMN notes TEXT")

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
    def _migrate_add_summary_is_final():
        """给已存在的 summaries 表补 is_final 列。"""
        engine = DatabaseManager._engine
        if engine is None:
            return
        with engine.begin() as connection:
            if not DatabaseManager._table_has_column(connection, "summaries", "is_final"):
                connection.exec_driver_sql(
                    "ALTER TABLE summaries ADD COLUMN is_final BOOLEAN DEFAULT 0"
                )

    @staticmethod
    def _migrate_create_fts_index():
        """建立 summaries 的全文索引（FTS5 + trigram）并用触发器保持同步。

        用 trigram 而不是默认的 unicode61：unicode61 按空白/标点切词，中文
        整句会变成一个 token，等于搜不了。trigram 切成 3 字符滑动窗口，中文
        和子串（错拼、词的一部分）都能匹配，代价是索引更大、且查询短于 3
        个字符时无法命中（调用方需回退到纯向量检索）。

        rowid 直接对齐 summaries.id，查询结果无需再做一次映射。
        """
        engine = DatabaseManager._engine
        if engine is None:
            return

        with engine.begin() as connection:
            connection.exec_driver_sql(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts
                USING fts5(content, tokenize='trigram')
                """
            )

            # 摘要由后台任务写入，靠应用层同步迟早会漏；交给触发器兜住
            connection.exec_driver_sql(
                """
                CREATE TRIGGER IF NOT EXISTS summaries_fts_ai AFTER INSERT ON summaries BEGIN
                    INSERT INTO summaries_fts(rowid, content) VALUES (new.id, new.content);
                END
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TRIGGER IF NOT EXISTS summaries_fts_ad AFTER DELETE ON summaries BEGIN
                    DELETE FROM summaries_fts WHERE rowid = old.id;
                END
                """
            )
            connection.exec_driver_sql(
                """
                CREATE TRIGGER IF NOT EXISTS summaries_fts_au AFTER UPDATE ON summaries BEGIN
                    DELETE FROM summaries_fts WHERE rowid = old.id;
                    INSERT INTO summaries_fts(rowid, content) VALUES (new.id, new.content);
                END
                """
            )

            # 回填历史数据（触发器只管建表之后的写入）
            missing = connection.exec_driver_sql(
                """
                SELECT count(*) FROM summaries
                WHERE id NOT IN (SELECT rowid FROM summaries_fts)
                """
            ).scalar()
            if missing:
                connection.exec_driver_sql(
                    """
                    INSERT INTO summaries_fts(rowid, content)
                    SELECT id, content FROM summaries
                    WHERE id NOT IN (SELECT rowid FROM summaries_fts)
                    """
                )

    @staticmethod
    def get_db() -> DBSession:
        if DatabaseManager._SessionLocal is None:
            DatabaseManager._init_db()
        return DatabaseManager._SessionLocal()

    @staticmethod
    def _init_db():
        if DatabaseManager._engine is None:
            db_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "transcripts.db"
            DatabaseManager._engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}
            )
            DatabaseManager._SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=DatabaseManager._engine
            )
            Base.metadata.create_all(bind=DatabaseManager._engine)
            DatabaseManager._migrate_add_session_name_notes()
            DatabaseManager._migrate_remove_redundant_project_columns()
            DatabaseManager._migrate_add_summary_is_final()
            DatabaseManager._migrate_create_fts_index()

    # ── Project ──────────────────────────────────────────────────────────────

    @staticmethod
    def create_project(name: str, description: str = "") -> Project:
        db = DatabaseManager.get_db()
        try:
            project = Project(name=name, description=description)
            db.add(project)
            db.commit()
            db.refresh(project)
            return project
        finally:
            db.close()

    @staticmethod
    def get_all_projects() -> List[Project]:
        db = DatabaseManager.get_db()
        try:
            rows = (
                db.query(Project, func.count(Session.id).label("session_count"))
                .outerjoin(Session, Session.project_id == Project.id)
                .group_by(Project.id)
                .order_by(Project.updated_at.desc())
                .all()
            )
            projects = []
            for project, session_count in rows:
                project.session_count = session_count
                projects.append(project)
            return projects
        finally:
            db.close()

    @staticmethod
    def get_project_by_id(project_id: int) -> Optional[Project]:
        db = DatabaseManager.get_db()
        try:
            row = (
                db.query(Project, func.count(Session.id).label("session_count"))
                .outerjoin(Session, Session.project_id == Project.id)
                .filter(Project.id == project_id)
                .group_by(Project.id)
                .first()
            )
            if row is None:
                return None
            project, session_count = row
            project.session_count = session_count
            return project
        finally:
            db.close()

    @staticmethod
    def delete_project(project_id: int) -> bool:
        db = DatabaseManager.get_db()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                db.delete(project)
                db.commit()
                # FAISS 索引不在数据库里，级联删不到它。留着的话磁盘上会堆积
                # 孤儿索引，全局检索遍历时还会搜出已删除项目的内容。
                try:
                    from faiss_manager import get_faiss_manager
                    get_faiss_manager().reset_index(project_id)
                except Exception as exc:
                    log.warning("Failed to drop FAISS index for project %d: %s", project_id, exc)
                return True
            return False
        finally:
            db.close()

    # ── Session ──────────────────────────────────────────────────────────────

    @staticmethod
    def create_session(project_id: int, mode: str) -> Session:
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
        ended_at: Optional[datetime] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[Session]:
        db = DatabaseManager.get_db()
        try:
            session = db.query(Session).filter(Session.id == session_id).first()
            if session is None:
                return None
            if duration_seconds is not None:
                session.duration_seconds = duration_seconds
            if transcript_text is not None:
                session.transcript_text = transcript_text
            if sentence_count is not None:
                session.sentence_count = sentence_count
            if char_count is not None:
                session.char_count = char_count
            if ended_at is not None:
                session.ended_at = ended_at
            if name is not None:
                session.name = name
            if notes is not None:
                session.notes = notes
            db.commit()
            db.refresh(session)
            return session
        finally:
            db.close()

    @staticmethod
    def get_project_sessions(project_id: int) -> List[Session]:
        db = DatabaseManager.get_db()
        try:
            return (
                db.query(Session)
                .filter(Session.project_id == project_id)
                .order_by(Session.started_at.desc())
                .all()
            )
        finally:
            db.close()

    @staticmethod
    def get_session_by_id(session_id: int) -> Optional[Session]:
        db = DatabaseManager.get_db()
        try:
            return db.query(Session).filter(Session.id == session_id).first()
        finally:
            db.close()

    @staticmethod
    def delete_session(session_id: int) -> bool:
        db = DatabaseManager.get_db()
        try:
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

    # ── Summary ──────────────────────────────────────────────────────────────

    @staticmethod
    def create_summary(
        session_id: int,
        content: str,
        source_text: str,
        start_sentence_idx: int,
        end_sentence_idx: int,
        duration_seconds: int = 0,
        is_final: bool = False,
    ) -> Summary:
        db = DatabaseManager.get_db()
        try:
            summary = Summary(
                session_id=session_id,
                content=content,
                source_text=source_text,
                start_sentence_idx=start_sentence_idx,
                end_sentence_idx=end_sentence_idx,
                duration_seconds=duration_seconds,
                is_final=is_final,
                created_at=datetime.utcnow(),
            )
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
    def delete_final_summaries(session_id: int) -> List[int]:
        """删掉某个 session 的终版摘要，返回被删的 id。

        继续录音时调用：旧的终版摘要只覆盖前半段，会被 Stop 时生成的新
        终版摘要完全包含，留着就是冗余。
        """
        db = DatabaseManager.get_db()
        try:
            rows = (
                db.query(Summary)
                .filter(Summary.session_id == session_id, Summary.is_final == True)
                .all()
            )
            deleted = [row.id for row in rows]
            for row in rows:
                db.delete(row)
            if deleted:
                db.commit()
            return deleted
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get_session_summaries(session_id: int) -> List[Summary]:
        db = DatabaseManager.get_db()
        try:
            return (
                db.query(Summary)
                .filter(Summary.session_id == session_id)
                .order_by(Summary.created_at.asc())
                .all()
            )
        finally:
            db.close()

    @staticmethod
    def get_summary_by_id(summary_id: int) -> Optional[Summary]:
        db = DatabaseManager.get_db()
        try:
            return db.query(Summary).filter(Summary.id == summary_id).first()
        finally:
            db.close()

    @staticmethod
    def delete_summary(summary_id: int) -> bool:
        db = DatabaseManager.get_db()
        try:
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
