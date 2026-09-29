"""Hybrid strategy.

Content-based and collaborative models fail in opposite directions: content
handles a brand-new item with no interactions, collaborative handles a long-tail
user whose items share no words. Fusing them recovers both.

Two fusion modes are provided:

* `score`  - linear blend of min-max normalised scores. Fast, but sensitive to
  the relative scale of the two models.
* `rrf`    - Reciprocal Rank Fusion, `sum(1 / (k + rank))`. Scale-free and
  robust, so it is the default; it only needs the orderings, not the values.
"""

from __future__ import annotations

import numpy as np

from app.models.base import BaseRecommender, FitContext, ScoredItem, rank_scores
from app.models.collaborative import (
    CollaborativeRecommender,
    PopularityRecommender,
    minmax_rows,
)
from app.models.content_based import ContentBasedRecommender

RRF_K = 60.0


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(
        self,
        content: ContentBasedRecommender | None = None,
        collaborative: CollaborativeRecommender | None = None,
        popularity: PopularityRecommender | None = None,
        mode: str = "rrf",
        content_weight: float = 0.4,
        collab_weight: float = 0.5,
        popularity_weight: float = 0.1,
        rrf_k: float = RRF_K,
        candidate_depth: int = 100,
        **params,
    ) -> None:
        super().__init__(
            mode=mode,
            content_weight=content_weight,
            collab_weight=collab_weight,
            popularity_weight=popularity_weight,
            rrf_k=rrf_k,
            candidate_depth=candidate_depth,
            **params,
        )
        if mode not in {"score", "rrf"}:
            raise ValueError(f"unknown fusion mode {mode!r}; expected 'score' or 'rrf'")
        self.content = content
        self.collaborative = collaborative
        self.popularity = popularity

    # -- fitting ---------------------------------------------------------
    def fit(self, ctx: FitContext) -> HybridRecommender:
        if self.content is None:
            self.content = ContentBasedRecommender()
        if self.collaborative is None:
            self.collaborative = CollaborativeRecommender()
        if self.popularity is None:
            self.popularity = PopularityRecommender()

        self.content.fit(ctx)
        self.collaborative.fit(ctx)
        self.popularity.fit(ctx)
        self._interactions = ctx.interactions
        self._finalise(ctx)
        return self

    # -- scoring ---------------------------------------------------------
    def score(self, user_id: str) -> np.ndarray:
        self._check()
        parts = [
            (self.params["content_weight"], self.content.score(user_id)),
            (self.params["collab_weight"], self.collaborative.score(user_id)),
            (self.params["popularity_weight"], self.popularity.score(user_id)),
        ]
        parts = [(w, s) for w, s in parts if s is not None and w > 0]
        total_weight = sum(w for w, _ in parts) or 1.0
        blended = np.zeros(len(self._item_ids), dtype=np.float64)
        for weight, scores in parts:
            blended += (weight / total_weight) * minmax_rows(
                np.asarray(scores, dtype=np.float32).reshape(1, -1)
            ).ravel()
        return blended

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude: set[str] | None = None,
        candidates: set[str] | None = None,
        include_seen: bool = False,
    ) -> list[ScoredItem]:
        self._check()
        if self.params["mode"] == "rrf":
            return self._recommend_rrf(user_id, top_k, exclude, candidates, include_seen)
        return self._recommend_score(user_id, top_k, exclude, candidates, include_seen)

    def _recommend_score(
        self, user_id: str, top_k: int, exclude, candidates, include_seen
    ) -> list[ScoredItem]:
        seen = set(self._interactions.item_ids_for_cols(self._interactions.seen_items(user_id)))
        if not include_seen:
            exclude = (exclude or set()) | seen
        return rank_scores(
            self.score(user_id), self._item_ids, top_k,
            exclude=exclude, candidates=candidates, include_seen=include_seen, reason="hybrid",
        )

    def _recommend_rrf(
        self, user_id: str, top_k: int, exclude, candidates, include_seen
    ) -> list[ScoredItem]:
        depth = max(self.params["candidate_depth"], top_k)
        k = self.params["rrf_k"]
        total: dict[str, float] = {}

        members = [
            (self.params["content_weight"], self.content, "content"),
            (self.params["collab_weight"], self.collaborative, "collaborative"),
            (self.params["popularity_weight"], self.popularity, "popularity"),
        ]
        for weight, model, _label in members:
            if weight <= 0:
                continue
            ranked = model.recommend(
                user_id, top_k=depth, exclude=exclude, include_seen=include_seen
            )
            for item in ranked:
                total[item.product_id] = total.get(item.product_id, 0.0) + weight / (
                    k + item.rank
                )

        item_ids = list(total)
        scores = np.array([total[p] for p in item_ids], dtype=np.float64)
        if not item_ids:
            return []
        fused = rank_scores(
            scores, item_ids, min(top_k, len(item_ids)),
            exclude=exclude, candidates=candidates, include_seen=include_seen, reason="hybrid-rrf",
        )
        return fused

    def similar(
        self, product_id: str, top_k: int = 10, exclude_self: bool = True
    ) -> list[ScoredItem]:
        """Fuse the two 'more like this' views for consistency with the feed."""
        k = self.params["rrf_k"]
        total: dict[str, float] = {}
        for weight, model in (
            (self.params["content_weight"], self.content),
            (self.params["collab_weight"], self.collaborative),
        ):
            if weight <= 0 or model is None:
                continue
            for item in model.similar(product_id, top_k=top_k * 2, exclude_self=exclude_self):
                total[item.product_id] = total.get(item.product_id, 0.0) + weight / (k + item.rank)
        if not total:
            return []
        item_ids = list(total)
        return rank_scores(
            np.array([total[p] for p in item_ids]), item_ids, min(top_k, len(item_ids)),
            exclude={product_id} if exclude_self else None, reason="hybrid-similar",
        )
