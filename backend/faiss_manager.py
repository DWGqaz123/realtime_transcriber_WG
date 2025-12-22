"""
FAISS 索引管理器
管理向量索引的创建、存储、加载和搜索
"""

import os
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果"""
    summary_id: int
    distance: float
    similarity: float  # 1 - distance (余弦距离)


class FAISSIndexManager:
    """FAISS 索引管理器"""
    
    def __init__(self, dimension: int = 384, index_dir: Optional[Path] = None):
        """
        初始化 FAISS 索引管理器
        
        Args:
            dimension: 向量维度（默认 384，对应 MiniLM）
            index_dir: 索引存储目录
        """
        self.dimension = dimension
        
        # 设置索引存储目录
        if index_dir is None:
            self.index_dir = Path.home() / "Library" / "Application Support" / "RealtimeTranscriber" / "faiss_indices"
        else:
            self.index_dir = Path(index_dir)
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 项目索引字典 {project_id: faiss.Index}
        self.indices: Dict[int, faiss.Index] = {}
        
        # ID 映射字典 {project_id: {faiss_id: summary_id}}
        self.id_mappings: Dict[int, Dict[int, int]] = {}
        
        print(f"🗂️  FAISS Index Manager initialized")
        print(f"   Dimension: {dimension}")
        print(f"   Index directory: {self.index_dir}")
    
    def _get_index_path(self, project_id: int) -> Path:
        """获取项目的索引文件路径"""
        return self.index_dir / f"project_{project_id}.index"
    
    def _get_mapping_path(self, project_id: int) -> Path:
        """获取项目的 ID 映射文件路径"""
        return self.index_dir / f"project_{project_id}_mapping.pkl"
    
    def create_index(self, project_id: int) -> faiss.Index:
        """
        创建新的 FAISS 索引（使用余弦相似度）
        
        Args:
            project_id: 项目 ID
            
        Returns:
            faiss.Index: 新创建的索引
        """
        # 使用 IndexFlatIP (内积) 配合归一化向量 = 余弦相似度
        index = faiss.IndexFlatIP(self.dimension)
        
        self.indices[project_id] = index
        self.id_mappings[project_id] = {}
        
        print(f"✅ Created new FAISS index for project {project_id}")
        return index
    
    def load_index(self, project_id: int) -> Optional[faiss.Index]:
        """
        从磁盘加载索引
        
        Args:
            project_id: 项目 ID
            
        Returns:
            faiss.Index or None: 加载的索引，如果不存在则返回 None
        """
        index_path = self._get_index_path(project_id)
        mapping_path = self._get_mapping_path(project_id)
        
        if not index_path.exists():
            print(f"ℹ️  No existing index found for project {project_id}")
            return None
        
        try:
            # 加载索引
            index = faiss.read_index(str(index_path))
            self.indices[project_id] = index
            
            # 加载 ID 映射
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    self.id_mappings[project_id] = pickle.load(f)
            else:
                self.id_mappings[project_id] = {}
            
            print(f"✅ Loaded index for project {project_id}")
            print(f"   Vectors: {index.ntotal}")
            print(f"   Mappings: {len(self.id_mappings[project_id])}")
            
            return index
            
        except Exception as e:
            print(f"❌ Failed to load index for project {project_id}: {e}")
            return None
    
    def save_index(self, project_id: int) -> bool:
        """
        保存索引到磁盘
        
        Args:
            project_id: 项目 ID
            
        Returns:
            bool: 是否保存成功
        """
        if project_id not in self.indices:
            print(f"⚠️  No index found for project {project_id}")
            return False
        
        try:
            index_path = self._get_index_path(project_id)
            mapping_path = self._get_mapping_path(project_id)
            
            # 保存索引
            faiss.write_index(self.indices[project_id], str(index_path))
            
            # 保存 ID 映射
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.id_mappings[project_id], f)
            
            print(f"✅ Saved index for project {project_id}")
            print(f"   Path: {index_path}")
            print(f"   Vectors: {self.indices[project_id].ntotal}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save index for project {project_id}: {e}")
            return False
    
    def add_vectors(
        self,
        project_id: int,
        embeddings: np.ndarray,
        summary_ids: List[int]
    ) -> bool:
        """
        添加向量到索引
        
        Args:
            project_id: 项目 ID
            embeddings: 向量矩阵 (N, dimension)
            summary_ids: 对应的 summary ID 列表
            
        Returns:
            bool: 是否添加成功
        """
        # 确保索引存在
        if project_id not in self.indices:
            index = self.load_index(project_id)
            if index is None:
                index = self.create_index(project_id)
        
        index = self.indices[project_id]
        
        # 归一化向量（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        
        # 当前索引大小
        start_id = index.ntotal
        
        # 添加向量
        index.add(embeddings)
        
        # 更新 ID 映射
        mapping = self.id_mappings[project_id]
        for i, summary_id in enumerate(summary_ids):
            faiss_id = start_id + i
            mapping[faiss_id] = summary_id
        
        print(f"✅ Added {len(summary_ids)} vectors to project {project_id}")
        print(f"   Total vectors: {index.ntotal}")
        
        return True
    
    def search(
        self,
        project_id: int,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        搜索最相似的向量
        
        Args:
            project_id: 项目 ID
            query_embedding: 查询向量 (dimension,)
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 确保索引存在
        if project_id not in self.indices:
            index = self.load_index(project_id)
            if index is None:
                print(f"⚠️  No index for project {project_id}")
                return []
        
        index = self.indices[project_id]
        
        if index.ntotal == 0:
            print(f"⚠️  Empty index for project {project_id}")
            return []
        
        # 归一化查询向量
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # 搜索
        distances, indices = index.search(query, min(top_k, index.ntotal))
        
        # 构建结果
        results = []
        mapping = self.id_mappings[project_id]
        
        for i, (distance, faiss_id) in enumerate(zip(distances[0], indices[0])):
            if faiss_id == -1:  # FAISS 返回 -1 表示无效结果
                continue
            
            summary_id = mapping.get(faiss_id)
            if summary_id is None:
                continue
            
            # 内积距离转换为相似度 (越大越相似)
            similarity = float(distance)
            
            results.append(SearchResult(
                summary_id=summary_id,
                distance=float(distance),
                similarity=similarity
            ))
        
        print(f"🔍 Search in project {project_id}")
        print(f"   Found {len(results)} results")
        
        return results
    
    def get_index(self, project_id: int) -> Optional[faiss.Index]:
        """获取项目的索引（自动加载）"""
        if project_id not in self.indices:
            return self.load_index(project_id)
        return self.indices[project_id]


# 全局单例
_faiss_manager: Optional[FAISSIndexManager] = None


def get_faiss_manager() -> FAISSIndexManager:
    """获取全局 FAISS 管理器实例"""
    global _faiss_manager
    
    if _faiss_manager is None:
        _faiss_manager = FAISSIndexManager(dimension=384)
    
    return _faiss_manager