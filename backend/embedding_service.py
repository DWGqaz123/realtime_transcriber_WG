"""
向量化服务
使用 Hugging Face Inference API 进行远程向量化
"""

from typing import List, Optional
import httpx
import numpy as np

from config import EmbeddingConfig


class EmbeddingService:
    """向量化服务（使用 Hugging Face Inference API）"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EmbeddingConfig.MODEL
        self.dimension = EmbeddingConfig.DIMENSION
        self.api_timeout = EmbeddingConfig.TIMEOUT_SECONDS
        self.batch_size = EmbeddingConfig.BATCH_SIZE
        self.api_url = EmbeddingConfig.get_api_url(self.model_name)
        self.api_key = EmbeddingConfig.get_api_key()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_vector(self, raw: object) -> np.ndarray:
        """把 Hugging Face 响应结构统一转换为 1D 向量。"""
        if not isinstance(raw, list) or not raw:
            raise ValueError("Invalid embedding response format")

        # 1D: [d]
        if isinstance(raw[0], (float, int)):
            vec = np.array(raw, dtype=np.float32)
            return vec

        # 2D token-level: [tokens, d] -> mean pooling
        if isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], (float, int)):
            token_matrix = np.array(raw, dtype=np.float32)
            return token_matrix.mean(axis=0)

        raise ValueError("Unsupported embedding shape from API")

    def _request_embeddings(self, inputs: List[str]) -> List[np.ndarray]:
        payload = {
            "inputs": inputs,
            "options": {"wait_for_model": True}
        }

        with httpx.Client(timeout=self.api_timeout) as client:
            response = client.post(self.api_url, headers=self._headers(), json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"Embedding API failed: {response.status_code} {response.text[:200]}")

        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Invalid embedding API response")

        # 单条输入时，可能返回 [d] 或 [tokens, d]
        if len(inputs) == 1:
            return [self._extract_vector(data)]

        # 批量输入时，期望 [N, ...]
        vectors: List[np.ndarray] = []
        for item in data:
            vectors.append(self._extract_vector(item))
        return vectors
    
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
        
        vectors = self._request_embeddings([text])
        vector = vectors[0]
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
            )
        return vector
    
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
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

        effective_batch = batch_size or self.batch_size
        all_vectors: List[np.ndarray] = []

        for i in range(0, len(valid_texts), effective_batch):
            chunk = valid_texts[i:i + effective_batch]
            all_vectors.extend(self._request_embeddings(chunk))

        embeddings = np.vstack(all_vectors).astype(np.float32)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )
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