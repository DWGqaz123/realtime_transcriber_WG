# backend/database/models.py

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Project(Base):
    """项目表"""
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    sessions = relationship("Session", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}')>"


class Session(Base):
    """录音会话表"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    mode = Column(String(50), nullable=False)  # "lecture" or "discussion"
    duration_seconds = Column(Integer, default=0)
    transcript_text = Column(Text, nullable=True)
    sentence_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    
    project = relationship("Project", back_populates="sessions")
    summaries = relationship("Summary", back_populates="session", cascade="all, delete-orphan")
     
    
    def __repr__(self):
        return f"<Session(id={self.id}, project_id={self.project_id}, mode='{self.mode}')>"

class Summary(Base):
    """摘要表"""
    __tablename__ = "summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    source_text = Column(Text)
    start_sentence_idx = Column(Integer)
    end_sentence_idx = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Integer)

    session = relationship("Session", back_populates="summaries")
    embedding = relationship("Embedding", back_populates="summary", uselist=False, cascade="all, delete-orphan")
    is_indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime)

    def __repr__(self):
        return f"<Summary(id={self.id}, session_id={self.session_id}, created_at={self.created_at})>"


class Embedding(Base):
    """向量嵌入表（存储元数据，向量存在 FAISS）"""
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    summary_id = Column(Integer, ForeignKey("summaries.id", ondelete="CASCADE"), unique=True, nullable=False)
    faiss_index_id = Column(Integer, nullable=False)
    content_preview = Column(Text)
    session_mode = Column(String(50))
    indexed_at = Column(DateTime, default=datetime.utcnow)
    embedding_model = Column(String(100), default="paraphrase-multilingual-MiniLM-L12-v2")
    embedding_dimension = Column(Integer, default=384)
    
    summary = relationship("Summary", back_populates="embedding")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, summary_id={self.summary_id}, faiss_id={self.faiss_index_id})>"
