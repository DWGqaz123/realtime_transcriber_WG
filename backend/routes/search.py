"""
搜索路由 - 语义搜索 API
"""

import asyncio

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from embedding_service import get_embedding_service
from faiss_manager import get_faiss_manager
from indexing_service import get_indexing_service
from hybrid_search import fuse, keyword_search
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
    similarity: float           # 余弦相似度；仅关键词命中时为 0
    session_id: int
    session_mode: str
    created_at: str
    score: float = 0.0          # RRF 融合分，决定排序
    bm25: Optional[float] = None
    sources: List[str] = []     # semantic / keyword，标明这条是哪一路捞到的

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """搜索响应"""
    query: str
    total: int
    results: List[SearchResult]
    mode: str = "hybrid"
    semantic_hits: int = 0      # 两路各自召回多少，便于判断检索行为
    keyword_hits: int = 0


# ==================== API 端点 ====================

@router.get("/projects/{project_id}", response_model=SearchResponse)
async def search_in_project(
    project_id: int,
    query: str = Query(..., min_length=1, description="搜索查询"),
    top_k: int = Query(10, ge=1, le=50, description="返回结果数量"),
    mode: str = Query("hybrid", pattern="^(hybrid|semantic|keyword)$", description="检索模式"),
    min_similarity: float = Query(0.15, ge=0.0, le=1.0, description="向量路的相似度下限"),
):
    """在项目中检索摘要。

    默认走混合检索：稠密向量负责同义改写与语义相近，BM25 负责精确术语、
    错拼和缩写，两路结果用 RRF 按名次融合。mode 可以强制走单路，用于
    对比或排查。
    """
    try:
        require_project(project_id)

        # 融合前每路都要多召回一些，否则排在两路各自 top_k 边界外、
        # 但综合排名靠前的条目会被提前丢掉
        recall_k = min(max(top_k * 3, 20), 100)

        vector_hits = []
        keyword_hits = []

        db = DatabaseManager.get_db()
        try:
            if mode in ("hybrid", "semantic"):
                embedding_service = get_embedding_service()
                query_embedding = await asyncio.to_thread(embedding_service.embed_text, query)
                faiss_manager = get_faiss_manager()
                vector_hits = [
                    (r.summary_id, r.similarity)
                    for r in faiss_manager.search(
                        project_id=project_id,
                        query_embedding=query_embedding,
                        top_k=recall_k,
                        min_similarity=min_similarity,
                    )
                ]

            if mode in ("hybrid", "keyword"):
                keyword_hits = await asyncio.to_thread(
                    keyword_search, db, project_id, query, recall_k
                )

            fused = fuse(vector_hits, keyword_hits, top_k=top_k)

            if not fused:
                return SearchResponse(
                    query=query, total=0, results=[], mode=mode,
                    semantic_hits=len(vector_hits), keyword_hits=len(keyword_hits),
                )

            summary_ids = [h.summary_id for h in fused]
            summary_rows = db.query(Summary, DBSession.mode).join(
                DBSession, DBSession.id == Summary.session_id
            ).filter(Summary.id.in_(summary_ids)).all()

            summaries_by_id = {
                summary.id: (summary, session_mode)
                for summary, session_mode in summary_rows
            }

            results = []
            for hit in fused:
                row = summaries_by_id.get(hit.summary_id)
                if row is None:
                    continue
                summary, session_mode = row
                results.append(SearchResult(
                    summary_id=summary.id,
                    content=summary.content,
                    similarity=hit.similarity if hit.similarity is not None else 0.0,
                    session_id=summary.session_id,
                    session_mode=session_mode,
                    created_at=summary.created_at.isoformat() if summary.created_at else "",
                    score=hit.score,
                    bm25=hit.bm25,
                    sources=hit.sources,
                ))
        finally:
            db.close()

        return SearchResponse(
            query=query,
            total=len(results),
            results=results,
            mode=mode,
            semantic_hits=len(vector_hits),
            keyword_hits=len(keyword_hits),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/projects/{project_id}/reindex")
async def reindex_project(project_id: int):
    """清空并重建项目的向量索引（换 embedding 模型或修复索引后调用）。"""
    require_project(project_id)
    result = await get_indexing_service().reindex_project(project_id)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Reindex failed"))
    return result


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
