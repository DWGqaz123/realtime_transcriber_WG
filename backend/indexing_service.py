"""
索引服务
负责将摘要向量化并存储到 FAISS 索引
"""

import asyncio
from typing import List, Optional
import numpy as np
from datetime import datetime

from database.db import DatabaseManager
from database.models import Summary, Embedding
from embedding_service import get_embedding_service
from faiss_manager import get_faiss_manager


class IndexingService:
    """索引服务 - 管理摘要的向量化和索引"""
    
    def __init__(self):
        """初始化索引服务"""
        self.embedding_service = get_embedding_service()
        self.faiss_manager = get_faiss_manager()
        
        print("📇 IndexingService initialized")
    
    async def index_session_summaries(self, session_id: int) -> dict:
        """
        索引某个 session 的所有摘要
        
        Args:
            session_id: 数据库 session ID
            
        Returns:
            dict: 索引结果统计
        """
        print(f"\n{'='*60}")
        print(f"📇 Indexing session {session_id}")
        print(f"{'='*60}\n")
        
        # 获取数据库会话
        db = DatabaseManager.get_db()
        
        try:
            # Step 1: 获取 session 信息
            from database.models import Session as DBSession
            session = db.query(DBSession).filter(DBSession.id == session_id).first()
            
            if not session:
                print(f"❌ Session {session_id} not found")
                return {"success": False, "error": "Session not found"}
            
            project_id = session.project_id
            print(f"📁 Project ID: {project_id}")
            
            # Step 2: 获取未索引的摘要
            summaries = db.query(Summary).filter(
                Summary.session_id == session_id,
                Summary.is_indexed == False
            ).all()
            
            if not summaries:
                print("ℹ️  No summaries to index")
                return {"success": True, "indexed": 0, "message": "No summaries to index"}
            
            print(f"📝 Found {len(summaries)} summaries to index")
            
            # Step 3: 批量向量化
            texts = [s.content for s in summaries]
            summary_ids = [s.id for s in summaries]
            
            print(f"\n🔄 Vectorizing {len(texts)} summaries...")
            embeddings = self.embedding_service.embed_batch(texts)
            
            # Step 4: 添加到 FAISS 索引
            print(f"\n📊 Adding to FAISS index...")
            success = self.faiss_manager.add_vectors(
                project_id=project_id,
                embeddings=embeddings,
                summary_ids=summary_ids
            )
            
            if not success:
                return {"success": False, "error": "Failed to add vectors to FAISS"}
            
            # Step 5: 保存 FAISS 索引到磁盘
            print(f"\n💾 Saving FAISS index...")
            self.faiss_manager.save_index(project_id)
            
            # Step 6: 保存元数据到数据库
            print(f"\n💾 Saving metadata to database...")
            
            # 获取 FAISS ID 映射
            mapping = self.faiss_manager.id_mappings[project_id]
            
            for i, summary in enumerate(summaries):
                # 找到对应的 FAISS ID
                faiss_id = None
                for fid, sid in mapping.items():
                    if sid == summary.id:
                        faiss_id = fid
                        break
                
                if faiss_id is None:
                    print(f"⚠️  No FAISS ID found for summary {summary.id}")
                    continue
                
                # 创建 Embedding 记录
                embedding = Embedding(
                    summary_id=summary.id,
                    project_id=project_id,
                    faiss_index_id=faiss_id,
                    content_preview=summary.content[:200],
                    session_mode=session.mode,
                    embedding_model=self.embedding_service.model_name,
                    embedding_dimension=self.embedding_service.dimension
                )
                
                db.add(embedding)
                
                # 更新 summary 的索引状态
                summary.is_indexed = True
                summary.indexed_at = datetime.utcnow()
                
                print(f"  ✅ Summary {summary.id} → FAISS ID {faiss_id}")
            
            db.commit()
            
            print(f"\n{'='*60}")
            print(f"✅ Indexing complete!")
            print(f"   Session: {session_id}")
            print(f"   Project: {project_id}")
            print(f"   Summaries indexed: {len(summaries)}")
            print(f"{'='*60}\n")
            
            return {
                "success": True,
                "indexed": len(summaries),
                "session_id": session_id,
                "project_id": project_id
            }
            
        except Exception as e:
            print(f"\n❌ Indexing failed: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()
            
            return {"success": False, "error": str(e)}
            
        finally:
            db.close()
    
    async def index_project_summaries(self, project_id: int) -> dict:
        """
        索引某个项目的所有未索引摘要
        
        Args:
            project_id: 项目 ID
            
        Returns:
            dict: 索引结果统计
        """
        print(f"\n{'='*60}")
        print(f"📇 Indexing all summaries in project {project_id}")
        print(f"{'='*60}\n")
        
        db = DatabaseManager.get_db()
        
        try:
            # 获取所有未索引的摘要
            summaries = db.query(Summary).filter(
                Summary.project_id == project_id,
                Summary.is_indexed == False
            ).all()
            
            if not summaries:
                print("ℹ️  No summaries to index")
                return {"success": True, "indexed": 0}
            
            print(f"📝 Found {len(summaries)} summaries to index")
            
            # 按 session 分组索引
            session_ids = list(set(s.session_id for s in summaries))
            print(f"📋 Across {len(session_ids)} sessions")
            
            total_indexed = 0
            
            for session_id in session_ids:
                result = await self.index_session_summaries(session_id)
                if result["success"]:
                    total_indexed += result.get("indexed", 0)
            
            print(f"\n{'='*60}")
            print(f"✅ Project indexing complete!")
            print(f"   Project: {project_id}")
            print(f"   Total summaries indexed: {total_indexed}")
            print(f"{'='*60}\n")
            
            return {
                "success": True,
                "indexed": total_indexed,
                "project_id": project_id
            }
            
        except Exception as e:
            print(f"\n❌ Project indexing failed: {e}")
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