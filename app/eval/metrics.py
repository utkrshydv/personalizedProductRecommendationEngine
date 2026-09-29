"""Offline ranking metrics.

Implemented from the standard definitions so the numbers are comparable with
anything else in the literature:

* Precision@k / Recall@k  - hit quality and hit coverage
* MAP@k                  - mean average precision over relevant items
* NDCG@k                 - rank-discounted gain, the usual headline metric
* MRR / HitRate@k        - "did we get at least one good item, and how high"
* Catalog coverage       - share of the catalogue ever recommended
* Novelty / Personalisation - how surprising, and how user-specific

All functions take a *single* user's ranked list and the user's ground-truth
set, so they compose cleanly into an evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse

EPS = 1e-12


def _as_list(recommended: Iterable[str] | None) -> list[str]:
    if recommended is None:
        return []
    return [str(p) for p in recommended]


def precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = _as_list(recommended)[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if p in relevant) / float(k)


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the relevant items that appear in the top k.

    Counts *distinct* relevant items, so a repeated id cannot push recall above
    1.0 and make a broken ranking look better than a working one.
    """
    if not relevant:
        return 0.0
    top = set(_as_list(recommended)[:k])
    return len(top & relevant) / float(len(relevant))


def hit_rate_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    top = set(_as_list(recommended)[:k])
    return 1.0 if top & relevant else 0.0


def average_precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """AP@k with the standard 'precision at each relevant hit' formulation."""
    if not relevant:
        return 0.0
    top = _as_list(recommended)[:k]
    hits = 0
    total = 0.0
    for rank, pid in enumerate(top, start=1):
        if pid in relevant:
            hits += 1
            total += hits / float(rank)
    return total / float(min(len(relevant), k))


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """NDCG@k with binary relevance and log2(rank+1) discount."""
    if not relevant:
        return 0.0
    top = _as_list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, pid in enumerate(top, start=1)
        if pid in relevant
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, pid in enumerate(_as_list(recommended)[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def coverage(all_recommendations: Iterable[Sequence[str]], catalog: set[str]) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    recommended: set[str] = set()
    for row in all_recommendations:
        recommended.update(_as_list(row))
    if not catalog:
        return 0.0
    return len(recommended & catalog) / float(len(catalog))


def novelty(recommended: Sequence[str], item_popularity: dict[str, float]) -> float:
    """Mean self-information -log2 p(item). Higher = less mainstream."""
    top = [p for p in _as_list(recommended) if p in item_popularity]
    if not top:
        return 0.0
    probs = [max(item_popularity[p], EPS) for p in top]
    return float(np.mean([-math.log2(p) for p in probs]))


def personalisation(all_recommendations: Sequence[Sequence[str]]) -> float:
    """1 - mean pairwise Jaccard overlap. 1.0 = every user gets a unique list."""
    rows = [set(_as_list(r)) for r in all_recommendations if len(r) > 1]
    if len(rows) < 2:
        return 0.0
    # Cap the pair count so this stays O(n) for large evaluation runs.
    step = max(1, len(rows) // 200)
    sample = rows[::step][:200]
    overlaps: list[float] = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            a, b = sample[i], sample[j]
            union = a | b
            overlaps.append(len(a & b) / len(union) if union else 0.0)
    return 1.0 - float(np.mean(overlaps))


def intra_list_diversity(
    recommended: Sequence[str],
    similarity: sparse.csr_matrix | None,
    item_index: dict[str, int] | None = None,
) -> float:
    """1 - mean pairwise similarity. Requires a precomputed item-item matrix."""
    top = [p for p in _as_list(recommended) if item_index and p in item_index]
    if similarity is None or item_index is None or len(top) < 2:
        return 0.0
    idx = np.array([item_index[p] for p in top], dtype=int)
    sub = similarity[idx][:, idx]
    n = idx.size
    dense = sub.toarray()
    total = float(dense.sum()) - float(np.trace(dense))
    return 1.0 - total / (n * (n - 1))
