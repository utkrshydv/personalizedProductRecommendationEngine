"""Common model interface and ranking helpers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from app.features.build import ItemFeatureBuilder
from app.features.interactions import InteractionMatrix

NEG_INF = -1e9


@dataclass
class FitContext:
    """Everything a recommender is allowed to look at while fitting."""

    products: pd.DataFrame
    interactions: InteractionMatrix
    features: ItemFeatureBuilder | None = None
    item_matrix: sparse.csr_matrix | None = None
    product_ids: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        products: pd.DataFrame,
        interactions: InteractionMatrix,
        features: ItemFeatureBuilder | None = None,
        **params,
    ) -> FitContext:
        item_ids = interactions.item_ids or products["product_id"].astype(str).tolist()
        return cls(
            products=products,
            interactions=interactions,
            features=features,
            product_ids=list(item_ids),
            params=params,
        )


@dataclass
class ScoredItem:
    product_id: str
    score: float
    rank: int = 0
    reason: str = ""


class BaseRecommender(abc.ABC):
    """Every strategy implements fit/recommend and shares the same signature."""

    name: str = "base"

    def __init__(self, **params) -> None:
        self.params = params
        self.fitted = False
        self._products: pd.DataFrame = pd.DataFrame()
        self._item_ids: list[str] = []

    # -- contract --------------------------------------------------------
    @abc.abstractmethod
    def fit(self, ctx: FitContext) -> BaseRecommender: ...

    @abc.abstractmethod
    def score(self, user_id: str) -> np.ndarray:
        """Dense score vector over the fitted catalogue, indexed like _item_ids."""

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude: set[str] | None = None,
        candidates: set[str] | None = None,
        include_seen: bool = False,
    ) -> list[ScoredItem]:
        scores = self.score(user_id)
        if scores is None or not np.isfinite(scores).any():
            return []
        return rank_scores(
            scores,
            self._item_ids,
            top_k,
            exclude=exclude,
            candidates=candidates,
            include_seen=include_seen,
        )

    def recommend_many(
        self,
        user_ids: list[str],
        top_k: int = 10,
        exclude: dict[str, set[str]] | None = None,
    ) -> dict[str, list[ScoredItem]]:
        out: dict[str, list[ScoredItem]] = {}
        exclude = exclude or {}
        for uid in user_ids:
            out[uid] = self.recommend(uid, top_k=top_k, exclude=exclude.get(uid))
        return out

    # -- shared plumbing --------------------------------------------------
    def _finalise(self, ctx: FitContext) -> None:
        self._products = ctx.products
        self._item_ids = list(ctx.product_ids)
        self.fitted = True

    # -- shapes / indices -------------------------------------------------
    @property
    def n_items(self) -> int:
        return len(self._item_ids)

    def _check(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before scoring")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, fitted={self.fitted})"


def rank_scores(
    scores: np.ndarray,
    item_ids: list[str],
    top_k: int,
    exclude: set[str] | None = None,
    candidates: set[str] | None = None,
    include_seen: bool = False,
    reason: str = "",
) -> list[ScoredItem]:
    """Mask disallowed items, then take the top-k deterministically."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    index_map = {p: i for i, p in enumerate(item_ids)}
    if exclude and not include_seen:
        for pid in exclude:
            idx = index_map.get(pid)
            if idx is not None:
                scores[idx] = NEG_INF
    if candidates is not None:
        allowed = np.zeros(len(item_ids), dtype=bool)
        for pid in candidates:
            idx = index_map.get(pid)
            if idx is not None:
                allowed[idx] = True
        scores[~allowed] = NEG_INF

    k = min(top_k, len(item_ids))
    if k <= 0:
        return []
    top = np.argpartition(-scores, kth=k - 1)[:k]
    # Deterministic: score desc, then product_id asc.
    order = sorted(top, key=lambda i: (-scores[i], item_ids[i]))
    return [
        ScoredItem(product_id=item_ids[i], score=float(scores[i]), rank=r + 1, reason=reason)
        for r, i in enumerate(order)
        if scores[i] > NEG_INF / 2
    ]
