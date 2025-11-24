# backend/database/models.py

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
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
    
    def __repr__(self):
        return f"<Session(id={self.id}, project_id={self.project_id}, mode='{self.mode}')>"