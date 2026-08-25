"""
索引服务
负责将摘要向量化并存储到 FAISS 索引
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from database.db import DatabaseManager
from database.models import Summary, Embedding
from embedding_service import get_embedding_service
from faiss_manager import get_faiss_manager

log = logging.getLogger("transcriber.indexing")


class IndexingService:
    """索引服务 - 管理摘要的向量化和索引"""
    
    def __init__(self):
        """初始化索引服务"""
        self.embedding_service = get_embedding_service()
        self.faiss_manager = get_faiss_manager()
        
    
    async def index_session_summaries(self, session_id: int) -> dict:
        """
        索引某个 session 的所有摘要
        
        Args:
            session_id: 数据库 session ID
            
        Returns:
            dict: 索引结果统计
        """
        
        # 获取数据库会话
        db = DatabaseManager.get_db()
        
        try:
            # Step 1: 获取 session 信息
            from database.models import Session as DBSession
            session = db.query(DBSession).filter(DBSession.id == session_id).first()
            
            if not session:
                return {"success": False, "error": "Session not found"}
            
            project_id = session.project_id
            
            # Step 2: 获取未索引的摘要
            summaries = db.query(Summary).filter(
                Summary.session_id == session_id,
                Summary.is_indexed == False
            ).all()
            
            if not summaries:
                return {"success": True, "indexed": 0, "message": "No summaries to index"}
            
            
            # Step 3: 批量向量化
            # 空内容会让 embed_batch 与 summary_ids 错位，这里先剔除
            indexable = [s for s in summaries if s.content and s.content.strip()]
            skipped = len(summaries) - len(indexable)
            if skipped:
                log.warning("Skipping %d summaries with empty content", skipped)
            if not indexable:
                return {"success": True, "indexed": 0, "message": "No indexable summaries"}

            summaries = indexable
            texts = [s.content for s in summaries]
            summary_ids = [s.id for s in summaries]

            # embed_batch 是同步 HTTP 调用，放到线程里以免阻塞事件循环
            embeddings = await asyncio.to_thread(self.embedding_service.embed_batch, texts)
            
            # Step 4: 添加到 FAISS 索引，直接获得分配的 faiss ID 列表
            assigned_faiss_ids = self.faiss_manager.add_vectors(
                project_id=project_id,
                embeddings=embeddings,
                summary_ids=summary_ids,
            )

            if not assigned_faiss_ids:
                return {"success": False, "error": "Failed to add vectors to FAISS"}

            # Step 5: 保存 FAISS 索引到磁盘
            self.faiss_manager.save_index(project_id)

            # Step 6: 保存元数据到数据库（直接用返回的 faiss ID，无需反转映射）
            for summary, faiss_id in zip(summaries, assigned_faiss_ids):
                db.add(Embedding(
                    summary_id=summary.id,
                    faiss_index_id=faiss_id,
                    content_preview=summary.content[:200],
                    session_mode=session.mode,
                    embedding_model=self.embedding_service.model_name,
                    embedding_dimension=self.embedding_service.dimension,
                ))
                summary.is_indexed = True
                summary.indexed_at = datetime.utcnow()
                
            
            db.commit()
            
            
            return {
                "success": True,
                "indexed": len(summaries),
                "session_id": session_id,
                "project_id": project_id
            }
            
        except Exception as e:
            db.rollback()
            
            return {"success": False, "error": str(e)}
            
        finally:
            db.close()
    
    async def reindex_project(self, project_id: int) -> dict:
        """清空并重建项目索引（换 embedding 模型后使用）。"""
        db = DatabaseManager.get_db()
        try:
            from database.models import Session as DBSession

            summary_ids = [
                row[0]
                for row in db.query(Summary.id)
                .join(DBSession)
                .filter(DBSession.project_id == project_id)
                .all()
            ]
            if summary_ids:
                db.query(Embedding).filter(Embedding.summary_id.in_(summary_ids)).delete(
                    synchronize_session=False
                )
                db.query(Summary).filter(Summary.id.in_(summary_ids)).update(
                    {Summary.is_indexed: False, Summary.indexed_at: None},
                    synchronize_session=False,
                )
                db.commit()
        except Exception as e:
            db.rollback()
            log.error("Failed to reset index metadata: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            db.close()

        self.faiss_manager.reset_index(project_id)
        return await self.index_project_summaries(project_id)

    async def index_project_summaries(self, project_id: int) -> dict:
        """
        索引某个项目的所有未索引摘要
        
        Args:
            project_id: 项目 ID
            
        Returns:
            dict: 索引结果统计
        """
        
        db = DatabaseManager.get_db()
        
        try:
            # 获取所有未索引的摘要
            from database.models import Session as DBSession

            summaries = db.query(Summary).join(DBSession).filter(
                DBSession.project_id == project_id,
                Summary.is_indexed == False
            ).all()
            
            if not summaries:
                return {"success": True, "indexed": 0}
            
            
            # 按 session 分组索引
            session_ids = list(set(s.session_id for s in summaries))
            
            total_indexed = 0
            
            for session_id in session_ids:
                result = await self.index_session_summaries(session_id)
                if result["success"]:
                    total_indexed += result.get("indexed", 0)
            
            
            return {
                "success": True,
                "indexed": total_indexed,
                "project_id": project_id
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
        finally:
            db.close()


# 全局单例
_indexing_service: Optional[IndexingService] = None


def get_indexing_service() -> IndexingService:
    """获取全局索引服务实例"""
    global _indexing_service
    
    if _indexing_service is None:
        _indexing_service = IndexingService()
    
    return _indexing_service
