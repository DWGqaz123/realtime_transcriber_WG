"""
向量化服务
使用 Sentence Transformers 进行本地向量化
"""

import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import time


class EmbeddingService:
    """向量化服务（使用 Sentence Transformers）"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化向量化服务
        
        Args:
            model_name: Sentence Transformers 模型名称
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # MiniLM 默认维度
        
        print(f"🤖 Initializing EmbeddingService")
        print(f"   Model: {model_name}")
        
        # 延迟加载模型（首次使用时加载）
        self._load_model()
    
    def _load_model(self):
        """加载 Sentence Transformers 模型"""
        if self.model is not None:
            return
        
        print(f"📦 Loading Sentence Transformers model: {self.model_name}")
        start_time = time.time()
        
        try:
            # 设置缓存目录
            cache_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 加载模型
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_dir
            )
            
            # 获取实际维度
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            elapsed = time.time() - start_time
            print(f"✅ Model loaded successfully in {elapsed:.2f}s")
            print(f"   Dimension: {self.dimension}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        单个文本向量化
        
        Args:
            text: 待向量化的文本
            
        Returns:
            np.ndarray: 向量（shape: [dimension]）
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # 确保模型已加载
        self._load_model()
        
        # 向量化
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量向量化
        
        Args:
            texts: 待向量化的文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 向量矩阵（shape: [len(texts), dimension]）
        """
        if not texts:
            return np.array([])
        
        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        # 确保模型已加载
        self._load_model()
        
        print(f"🔄 Embedding {len(valid_texts)} texts...")
        start_time = time.time()
        
        # 批量向量化
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Embedded {len(valid_texts)} texts in {elapsed:.2f}s")
        print(f"   Speed: {len(valid_texts) / elapsed:.1f} texts/sec")
        
        return embeddings
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension


# 全局单例
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取全局 EmbeddingService 实例"""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service