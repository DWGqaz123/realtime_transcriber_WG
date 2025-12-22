"""
搜索路由 - 语义搜索 API
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from embedding_service import get_embedding_service
from faiss_manager import get_faiss_manager
from database.db import DatabaseManager
from database.models import Summary, Session as DBSession

router = APIRouter(prefix="/api/search", tags=["search"])


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
    print(f"\n{'='*60}")
    print(f"🔍 Search Request")
    print(f"   Project: {project_id}")
    print(f"   Query: '{query}'")
    print(f"   Top K: {top_k}")
    print(f"{'='*60}\n")
    
    try:
        # Step 1: 验证项目存在
        project = DatabaseManager.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Step 2: 向量化查询
        embedding_service = get_embedding_service()
        print(f"🔄 Vectorizing query...")
        query_embedding = embedding_service.embed_text(query)
        print(f"✅ Query vectorized")
        
        # Step 3: FAISS 向量搜索
        faiss_manager = get_faiss_manager()
        print(f"🔍 Searching in FAISS index...")
        faiss_results = faiss_manager.search(
            project_id=project_id,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        if not faiss_results:
            print(f"ℹ️  No results found")
            return SearchResponse(
                query=query,
                total=0,
                results=[]
            )
        
        print(f"✅ Found {len(faiss_results)} results from FAISS")
        
        # Step 4: 从数据库加载完整信息
        db = DatabaseManager.get_db()
        results = []
        
        try:
            for faiss_result in faiss_results:
                # 查询 summary 及其关联的 session
                summary = db.query(Summary).filter(
                    Summary.id == faiss_result.summary_id
                ).first()
                
                if not summary:
                    print(f"⚠️  Summary {faiss_result.summary_id} not found in DB")
                    continue
                
                # 查询 session 信息
                session = db.query(DBSession).filter(
                    DBSession.id == summary.session_id
                ).first()
                
                if not session:
                    print(f"⚠️  Session {summary.session_id} not found in DB")
                    continue
                
                # 构建结果
                result = SearchResult(
                    summary_id=summary.id,
                    content=summary.content,
                    similarity=faiss_result.similarity,
                    session_id=summary.session_id,
                    session_mode=session.mode,
                    created_at=summary.created_at.isoformat() if summary.created_at else ""
                )
                results.append(result)
                
                print(f"  ✅ [{len(results)}] Summary {summary.id} (similarity: {faiss_result.similarity:.4f})")
            
        finally:
            db.close()
        
        print(f"\n✅ Search complete: {len(results)} results")
        print(f"{'='*60}\n")
        
        return SearchResponse(
            query=query,
            total=len(results),
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
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
        # 验证项目存在
        project = DatabaseManager.get_project_by_id(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # 获取 FAISS 索引信息
        faiss_manager = get_faiss_manager()
        index = faiss_manager.get_index(project_id)
        
        # 获取数据库统计
        db = DatabaseManager.get_db()
        try:
            from database.models import Embedding
            
            total_summaries = db.query(Summary).filter(
                Summary.project_id == project_id
            ).count()
            
            indexed_summaries = db.query(Summary).filter(
                Summary.project_id == project_id,
                Summary.is_indexed == True
            ).count()
            
            embeddings_count = db.query(Embedding).filter(
                Embedding.project_id == project_id
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
        print(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")