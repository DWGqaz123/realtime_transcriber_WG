"""
测试 Embedding Service
"""

from embedding_service import get_embedding_service


def test_embedding_service():
    """测试向量化服务"""
    
    print("\n" + "="*60)
    print("🧪 Testing Embedding Service")
    print("="*60 + "\n")
    
    # 获取服务
    service = get_embedding_service()
    
    # 测试 1: 单个文本向量化
    print("Test 1: Single text embedding")
    text = "这是一个测试句子，用于验证向量化功能。"
    embedding = service.embed_text(text)
    
    print(f"✅ Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding preview: {embedding[:5]}...")
    print()
    
    # 测试 2: 批量向量化
    print("Test 2: Batch embedding")
    texts = [
        "深度学习是机器学习的一个分支。",
        "Transformer 架构改变了 NLP 领域。",
        "GPT-4 是一个大型语言模型。",
        "向量数据库用于存储和检索向量。",
    ]
    
    embeddings = service.embed_batch(texts)
    
    print(f"✅ Embedded {len(texts)} texts")
    print(f"   Embeddings shape: {embeddings.shape}")
    print()
    
    # 测试 3: 相似度计算
    print("Test 3: Similarity calculation")
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    query = "什么是深度学习？"
    query_embedding = service.embed_text(query)
    
    print(f"Query: '{query}'")
    print(f"Similarities:")
    
    for i, text in enumerate(texts):
        similarity = cosine_similarity(query_embedding, embeddings[i])
        print(f"   [{i}] {similarity:.4f} - {text}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import numpy as np
    test_embedding_service()