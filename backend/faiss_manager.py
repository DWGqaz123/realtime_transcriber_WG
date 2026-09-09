"""
FAISS 索引管理器
管理向量索引的创建、存储、加载和搜索
"""

import faiss
import logging
import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from config import EmbeddingConfig

log = logging.getLogger("transcriber.faiss")


@dataclass
class SearchResult:
    summary_id: int
    similarity: float  # 内积 = 余弦相似度（向量已归一化）


class FAISSIndexManager:
    """FAISS 索引管理器"""
    
    def __init__(self, dimension: Optional[int] = None, index_dir: Optional[Path] = None):
        """
        初始化 FAISS 索引管理器
        
        Args:
            dimension: 向量维度（默认 384，对应 MiniLM）
            index_dir: 索引存储目录
        """
        self.dimension = dimension or EmbeddingConfig.DIMENSION

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
        
    
    def _get_index_path(self, project_id: int) -> Path:
        """获取项目的索引文件路径（含维度，避免换 embedding 模型后误读旧索引）"""
        return self.index_dir / f"project_{project_id}_d{self.dimension}.index"

    def _get_mapping_path(self, project_id: int) -> Path:
        """获取项目的 ID 映射文件路径"""
        return self.index_dir / f"project_{project_id}_d{self.dimension}_mapping.pkl"
    
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
            return None
        
        try:
            index = faiss.read_index(str(index_path))
            if index.d != self.dimension:
                log.warning(
                    "Ignoring FAISS index for project %d: dimension %d != expected %d",
                    project_id, index.d, self.dimension,
                )
                return None
            self.indices[project_id] = index
            if mapping_path.exists():
                with open(mapping_path, "rb") as f:
                    self.id_mappings[project_id] = pickle.load(f)
            else:
                self.id_mappings[project_id] = {}
            return index
        except Exception as e:
            log.warning("Failed to load FAISS index for project %d: %s", project_id, e)
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
            return False
        
        try:
            index_path = self._get_index_path(project_id)
            mapping_path = self._get_mapping_path(project_id)
            
            # 保存索引
            faiss.write_index(self.indices[project_id], str(index_path))
            
            # 保存 ID 映射
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.id_mappings[project_id], f)
            
            
            return True
            
        except Exception:
            return False
    
    def add_vectors(
        self,
        project_id: int,
        embeddings: np.ndarray,
        summary_ids: List[int],
    ) -> List[int]:
        """
        添加向量到索引，返回分配的 faiss ID 列表（与 summary_ids 一一对应）。
        失败时返回空列表。
        """
        if project_id not in self.indices:
            index = self.load_index(project_id)
            if index is None:
                index = self.create_index(project_id)

        index = self.indices[project_id]

        if embeddings.ndim != 2 or embeddings.shape[0] != len(summary_ids):
            log.error(
                "add_vectors shape mismatch: embeddings=%s, summary_ids=%d",
                getattr(embeddings, "shape", None), len(summary_ids),
            )
            return []

        faiss.normalize_L2(embeddings)

        start_id = index.ntotal
        index.add(embeddings)

        mapping = self.id_mappings[project_id]
        assigned_ids = []
        for i, summary_id in enumerate(summary_ids):
            faiss_id = start_id + i
            mapping[faiss_id] = summary_id
            assigned_ids.append(faiss_id)

        return assigned_ids
    
    def search(
        self,
        project_id: int,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """
        搜索最相似的向量
        
        Args:
            project_id: 项目 ID
            query_embedding: 查询向量 (dimension,)
            top_k: 返回结果数量
            min_similarity: 相似度下限。IndexFlatIP 永远会返回 top_k 条，
                哪怕全是噪音（实测无关查询也有 0.03 左右），不设下限的话
                结果列表尾部永远挂着一堆不相关内容。

        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 确保索引存在
        if project_id not in self.indices:
            index = self.load_index(project_id)
            if index is None:
                return []
        
        index = self.indices[project_id]
        
        if index.ntotal == 0:
            return []
        
        # 归一化查询向量
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # 搜索
        distances, indices = index.search(query, min(top_k, index.ntotal))
        
        # 构建结果
        results = []
        mapping = self.id_mappings[project_id]
        
        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue
            if float(score) < min_similarity:
                continue
            summary_id = mapping.get(faiss_id)
            if summary_id is not None:
                results.append(SearchResult(summary_id=summary_id, similarity=float(score)))
        
        
        return results
    
    def search_many(
        self,
        project_ids: List[int],
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """跨多个项目检索后合并排序。

        索引是按项目分片的（每个项目一个 IndexFlatIP），所以全局检索只能
        逐个搜再合并。summary_id 是数据库主键、全局唯一，合并不会冲突。
        项目数是个位数、每个索引也只有几千条，逐个暴力搜的开销可以忽略。

        调用方必须传入数据库中真实存在的 project_id：磁盘上可能残留已删除
        项目的索引文件，照着文件遍历会搜出幽灵数据。
        """
        merged: List[SearchResult] = []
        for project_id in project_ids:
            merged.extend(
                self.search(
                    project_id=project_id,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    min_similarity=min_similarity,
                )
            )
        merged.sort(key=lambda r: r.similarity, reverse=True)
        return merged[:top_k]

    def remove_mappings(self, project_id: int, summary_ids: List[int]) -> int:
        """把指向这些 summary 的映射从索引里摘掉，返回摘除条数。

        IndexFlatIP 不支持删除向量，所以向量本身会留在索引里成为悬空数据；
        但移除映射后检索不会再返回它。这一步是必要的：SQLite 会复用被删除
        的主键，新摘要拿到同一个 id 后，旧向量就会指向新内容，导致检索时
        用旧语义匹配却返回新文本。彻底清理需要重建索引。
        """
        mapping = self.id_mappings.get(project_id)
        if not mapping:
            return 0

        targets = set(summary_ids)
        stale = [faiss_id for faiss_id, summary_id in mapping.items() if summary_id in targets]
        for faiss_id in stale:
            mapping.pop(faiss_id, None)

        if stale:
            self.save_index(project_id)
            log.info("Removed %d stale mapping(s) from project %d index", len(stale), project_id)
        return len(stale)

    def reset_index(self, project_id: int) -> None:
        """丢弃项目的索引与映射（内存 + 磁盘），供重新索引使用。"""
        self.indices.pop(project_id, None)
        self.id_mappings.pop(project_id, None)
        for path in (self._get_index_path(project_id), self._get_mapping_path(project_id)):
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                log.warning("Failed to remove %s: %s", path, exc)
        log.info("Reset FAISS index for project %d", project_id)

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
        _faiss_manager = FAISSIndexManager(dimension=EmbeddingConfig.DIMENSION)
    
    return _faiss_manager
