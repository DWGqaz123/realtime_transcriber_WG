"""
向量化服务

默认调用 OpenAI embeddings（复用摘要用的 OPENAI_API_KEY）；
设 EMBEDDING_PROVIDER=huggingface 可切到 HF Inference。
"""

from typing import List, Optional
import logging

import httpx
import numpy as np

from config import EmbeddingConfig

log = logging.getLogger("transcriber.embedding")


class EmbeddingService:
    """向量化服务"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EmbeddingConfig.MODEL
        self.dimension = EmbeddingConfig.DIMENSION
        self.api_timeout = EmbeddingConfig.TIMEOUT_SECONDS
        self.batch_size = EmbeddingConfig.BATCH_SIZE
        self.is_openai = EmbeddingConfig.is_openai()
        self.api_url = EmbeddingConfig.get_api_url(self.model_name)

    @property
    def api_key(self) -> str:
        # 读成 property：后端重启后 key 才会变，但这样不会把空 key 固化在单例里
        return EmbeddingConfig.get_api_key()

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "No embedding API key configured. Set the OpenAI API key in Settings "
                "(or HUGGINGFACE_API_KEY when EMBEDDING_PROVIDER=huggingface)."
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _extract_vector(self, raw: object) -> np.ndarray:
        """把 Hugging Face 响应结构统一转换为 1D 向量。"""
        if not isinstance(raw, list) or not raw:
            raise ValueError("Invalid embedding response format")

        # 1D: [d]
        if isinstance(raw[0], (float, int)):
            return np.array(raw, dtype=np.float32)

        # 2D token-level: [tokens, d] -> mean pooling
        if isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], (float, int)):
            token_matrix = np.array(raw, dtype=np.float32)
            return token_matrix.mean(axis=0)

        raise ValueError("Unsupported embedding shape from API")

    def _parse_openai(self, data: object, expected: int) -> List[np.ndarray]:
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != expected:
            raise ValueError(f"Invalid OpenAI embedding response (expected {expected} vectors)")
        # 按 index 排序，保证与输入顺序严格一一对应
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [np.array(item["embedding"], dtype=np.float32) for item in ordered]

    def _parse_huggingface(self, data: object, expected: int) -> List[np.ndarray]:
        if not isinstance(data, list):
            raise ValueError("Invalid embedding API response")
        # 单条输入时可能返回 [d] 或 [tokens, d]
        if expected == 1:
            return [self._extract_vector(data)]
        if len(data) != expected:
            raise ValueError(f"Invalid embedding response (expected {expected} vectors)")
        return [self._extract_vector(item) for item in data]

    def _request_embeddings(self, inputs: List[str]) -> List[np.ndarray]:
        if self.is_openai:
            payload = {"model": self.model_name, "input": inputs}
        else:
            payload = {"inputs": inputs, "options": {"wait_for_model": True}}

        with httpx.Client(timeout=self.api_timeout) as client:
            response = client.post(self.api_url, headers=self._headers(), json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding API failed: {response.status_code} {response.text[:200]}"
            )

        data = response.json()
        if self.is_openai:
            return self._parse_openai(data, len(inputs))
        return self._parse_huggingface(data, len(inputs))

    def embed_text(self, text: str) -> np.ndarray:
        """
        单个文本向量化

        Returns:
            np.ndarray: 向量（shape: [dimension]）
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        vector = self._request_embeddings([text])[0]
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
            )
        return vector

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        批量向量化。输出行数与 texts 严格一一对应——调用方需要靠这个顺序
        把向量映射回 summary_id，所以这里不做静默过滤。

        Returns:
            np.ndarray: 向量矩阵（shape: [len(texts), dimension]）
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        empty_positions = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty_positions:
            raise ValueError(
                f"embed_batch received empty text at positions {empty_positions}; "
                "filter them out before calling"
            )

        effective_batch = batch_size or self.batch_size
        all_vectors: List[np.ndarray] = []

        for i in range(0, len(texts), effective_batch):
            chunk = texts[i:i + effective_batch]
            vectors = self._request_embeddings(chunk)
            if len(vectors) != len(chunk):
                raise ValueError(
                    f"Embedding count mismatch: sent {len(chunk)}, got {len(vectors)}"
                )
            all_vectors.extend(vectors)

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
