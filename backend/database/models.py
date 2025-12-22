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
    
    # 关系：一个项目有多个会话
    sessions = relationship("Session", back_populates="project", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}')>"


class Session(Base):
    """录音会话表"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    # 会话元数据
    mode = Column(String(50), nullable=False)  # "lecture" or "discussion"
    duration_seconds = Column(Integer, default=0)
    transcript_text = Column(Text, nullable=True)
    sentence_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    
    # 时间戳
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    
    # 关系：属于某个项目
    project = relationship("Project", back_populates="sessions")
    summaries = relationship("Summary", back_populates="session", cascade="all, delete-orphan") 
     
    
    def __repr__(self):
        return f"<Session(id={self.id}, project_id={self.project_id}, mode='{self.mode}')>"

class Summary(Base):
    """摘要表 - 存储 AI 生成的智能摘要"""
    __tablename__ = "summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)  # 🔧 关联项目
    
    # 摘要内容
    content = Column(Text, nullable=False)  # Markdown 格式的摘要
    source_text = Column(Text)  # 原始文本片段
    
    # 位置信息
    start_sentence_idx = Column(Integer)  # 从第几句开始
    end_sentence_idx = Column(Integer)    # 到第几句结束
    
    # 时间信息
    created_at = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Integer)  # 这段摘要覆盖的时长
    
    
    # 关系
    session = relationship("Session", back_populates="summaries")
    project = relationship("Project", back_populates="summaries")
    embedding = relationship("Embedding", back_populates="summary", uselist=False, cascade="all, delete-orphan")
    
    # 索引状态
    is_indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime)
    
    
    
    def __repr__(self):
        return f"<Summary(id={self.id}, session_id={self.session_id}, created_at={self.created_at})>"


class Embedding(Base):
    """向量嵌入表（存储元数据，向量存在 FAISS）"""
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # 关联信息
    summary_id = Column(Integer, ForeignKey("summaries.id", ondelete="CASCADE"), unique=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    
    # FAISS 索引信息
    faiss_index_id = Column(Integer, nullable=False)  # FAISS 内部 ID
    
    # 元数据（用于过滤和展示）
    content_preview = Column(Text)                     # 前 200 字符
    session_mode = Column(String(50))                  # lecture / discussion
    
    # 索引状态
    indexed_at = Column(DateTime, default=datetime.utcnow)
    embedding_model = Column(String(100), default="paraphrase-multilingual-MiniLM-L12-v2")
    embedding_dimension = Column(Integer, default=384)
    
    # 关系
    summary = relationship("Summary", back_populates="embedding")
    project = relationship("Project")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, summary_id={self.summary_id}, faiss_id={self.faiss_index_id})>"
