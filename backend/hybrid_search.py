"""混合检索：稠密向量 + BM25 关键词，用 RRF 融合。

单靠向量检索有三类查询很吃力：
  - 精确术语（"FAISS"、"IndexFlatIP"）会被归到笼统的语义邻域里
  - 错拼与识别错误（搜 "Carnegie Mellon"，库里存的是 "Carnation Melon"）
  - 人名、编号、缩写这类语义负载低的 token

BM25 正好在这些上面强，而它对同义改写无能为力——两者互补。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("transcriber.search")

# trigram tokenizer 的窗口大小：短于 3 个字符的词在索引里没有对应 token
MIN_TRIGRAM_LEN = 3

# RRF 的平滑常数。60 是 Cormack 等人原始论文里的取值，作用是压低头部
# 名次的绝对优势，让两路的第 1 名不会碾压另一路的第 2、3 名。
RRF_K = 60


@dataclass
class HybridResult:
    summary_id: int
    score: float                          # RRF 融合分，仅用于排序
    similarity: Optional[float] = None    # 余弦相似度，未命中向量路时为 None
    bm25: Optional[float] = None          # BM25 分，未命中关键词路时为 None
    sources: List[str] = field(default_factory=list)   # semantic / keyword


def build_fts_query(query: str) -> Optional[str]:
    """把用户输入转成 FTS5 MATCH 表达式；无可用 token 时返回 None。

    每个词都包成短语（trigram 下等价于子串匹配），再用 OR 连接：命中
    的词越多 BM25 分越高，但不要求全部命中。中文没有空格，整串会成为
    一个 token，靠 trigram 实现子串匹配。
    """
    # 去掉 FTS5 语法字符，避免用户输入被解释成布尔表达式
    cleaned = re.sub(r'[")(*:^-]', " ", query).strip()
    if not cleaned:
        return None

    tokens = [t for t in cleaned.split() if len(t) >= MIN_TRIGRAM_LEN]
    if not tokens:
        return None

    return " OR ".join(f'"{t}"' for t in tokens)


def keyword_search(
    db,
    project_id: int,
    query: str,
    limit: int = 50,
) -> List[Tuple[int, float]]:
    """BM25 检索，返回 [(summary_id, bm25_score)]，已按相关度排序。

    bm25() 返回负数且越小越相关，这里取负号翻成"越大越相关"，方便展示。
    """
    match_expr = build_fts_query(query)
    if match_expr is None:
        log.debug("Query %r has no token >= %d chars, skipping BM25", query, MIN_TRIGRAM_LEN)
        return []

    try:
        rows = db.execute(
            _KEYWORD_SQL,
            {"match": match_expr, "project_id": project_id, "limit": limit},
        ).fetchall()
    except Exception as exc:
        # FTS 表缺失或表达式非法都不该让整个搜索挂掉，降级成纯向量
        log.warning("BM25 search failed (%s), falling back to vector only", exc)
        return []

    return [(int(r[0]), -float(r[1])) for r in rows]


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[int]],
    weights: Optional[Sequence[float]] = None,
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """RRF：按名次而非原始分数融合。

    向量相似度（0~1）和 BM25（无上界）不同量纲，直接加权相加没有意义，
    还会被某一路的分数分布带偏。RRF 只看名次，天然免疫这个问题。

        score(d) = Σ_i  weight_i / (k + rank_i(d))
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: Dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fuse(
    vector_hits: Sequence[Tuple[int, float]],
    keyword_hits: Sequence[Tuple[int, float]],
    top_k: int,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> List[HybridResult]:
    """把两路结果融合成最终排序。入参均为 [(summary_id, score)]。"""
    sim_by_id = {sid: score for sid, score in vector_hits}
    bm25_by_id = {sid: score for sid, score in keyword_hits}

    fused = reciprocal_rank_fusion(
        [[sid for sid, _ in vector_hits], [sid for sid, _ in keyword_hits]],
        weights=[semantic_weight, keyword_weight],
    )

    results: List[HybridResult] = []
    for summary_id, score in fused[:top_k]:
        sources = []
        if summary_id in sim_by_id:
            sources.append("semantic")
        if summary_id in bm25_by_id:
            sources.append("keyword")
        results.append(
            HybridResult(
                summary_id=summary_id,
                score=score,
                similarity=sim_by_id.get(summary_id),
                bm25=bm25_by_id.get(summary_id),
                sources=sources,
            )
        )
    return results


# 放在模块末尾，避免上面的函数定义被一大段 SQL 打断
_KEYWORD_SQL = None


def _init_sql():
    global _KEYWORD_SQL
    from sqlalchemy import text
    _KEYWORD_SQL = text(
        """
        SELECT f.rowid AS summary_id, bm25(summaries_fts) AS score
        FROM summaries_fts f
        JOIN summaries s ON s.id = f.rowid
        JOIN sessions ss ON ss.id = s.session_id
        WHERE summaries_fts MATCH :match
          AND ss.project_id = :project_id
        ORDER BY score
        LIMIT :limit
        """
    )


_init_sql()
