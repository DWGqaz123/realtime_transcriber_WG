"""
搜索路由 - 语义搜索 API
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List
from pydantic import BaseModel

from embedding_service import get_embedding_service
from faiss_manager import get_faiss_manager
from database.db import DatabaseManager
from database.models import Summary, Session as DBSession, Embedding

router = APIRouter(prefix="/api/search", tags=["search"])


def require_project(project_id: int):
    project = DatabaseManager.get_project_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


# ==================== 数据模型 ====================

class SearchResult(BaseModel):
    """搜索结果"""
    summary_id: int
    content: str
    similarity: float
    session_id: int
    session_mode: str
    created_at: str
    
    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """搜索响应"""
    query: str
    total: int
    results: List[SearchResult]


# ==================== API 端点 ====================

@router.get("/projects/{project_id}", response_model=SearchResponse)
async def search_in_project(
    project_id: int,
    query: str = Query(..., min_length=1, description="搜索查询"),
    top_k: int = Query(10, ge=1, le=50, description="返回结果数量")
):
    """
    在项目中搜索相关摘要
    
    Args:
        project_id: 项目 ID
        query: 搜索查询文本
        top_k: 返回的结果数量（1-50）
        
    Returns:
        SearchResponse: 搜索结果
    """
    
    try:
        require_project(project_id)
        
        # Step 2: 向量化查询
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.embed_text(query)
        
        # Step 3: FAISS 向量搜索
        faiss_manager = get_faiss_manager()
        faiss_results = faiss_manager.search(
            project_id=project_id,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not faiss_results:
            return SearchResponse(
                query=query,
                total=0,
                results=[]
            )
        
        
        # Step 4: 从数据库加载完整信息
        db = DatabaseManager.get_db()
        results = []

        try:
            summary_ids = [item.summary_id for item in faiss_results]
            summary_rows = db.query(Summary, DBSession.mode).join(
                DBSession, DBSession.id == Summary.session_id
            ).filter(Summary.id.in_(summary_ids)).all()

            summaries_by_id = {
                summary.id: (summary, session_mode)
                for summary, session_mode in summary_rows
            }

            for faiss_result in faiss_results:
                row = summaries_by_id.get(faiss_result.summary_id)
                if row is None:
                    continue

                summary, session_mode = row
                results.append(SearchResult(
                    summary_id=summary.id,
                    content=summary.content,
                    similarity=faiss_result.similarity,
                    session_id=summary.session_id,
                    session_mode=session_mode,
                    created_at=summary.created_at.isoformat() if summary.created_at else ""
                ))
        finally:
            db.close()
        
        
        return SearchResponse(
            query=query,
            total=len(results),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/projects/{project_id}/stats")
async def get_search_stats(project_id: int):
    """
    获取项目的搜索统计信息
    
    Args:
        project_id: 项目 ID
        
    Returns:
        dict: 统计信息
    """
    try:
        project = require_project(project_id)
        
        # 获取 FAISS 索引信息
        faiss_manager = get_faiss_manager()
        index = faiss_manager.get_index(project_id)
        
        # 获取数据库统计
        db = DatabaseManager.get_db()
        try:
            total_summaries = db.query(Summary).join(DBSession).filter(
                DBSession.project_id == project_id
            ).count()
            
            indexed_summaries = db.query(Summary).join(DBSession).filter(
                DBSession.project_id == project_id,
                Summary.is_indexed == True
            ).count()
            
            embeddings_count = db.query(Embedding).join(Summary).join(DBSession).filter(
                DBSession.project_id == project_id
            ).count()
            
        finally:
            db.close()
        
        vector_count = index.ntotal if index else 0
        
        return {
            "project_id": project_id,
            "project_name": str(project.name),
            "total_summaries": total_summaries,
            "indexed_summaries": indexed_summaries,
            "embeddings_count": embeddings_count,
            "vector_count": vector_count,
            "index_exists": index is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
