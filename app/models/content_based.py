"""Content-based filtering.

A user's taste is represented as a single vector: the recency-weighted centroid
of the feature vectors of everything they have interacted with. Scoring is then
a cosine similarity between that profile and the catalogue.

Why a centroid rather than a full linear model: the profile is order-free,
computable in O(history), and — importantly for serving — it can be updated
incrementally the moment a new event arrives, without retraining.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from app.features.build import ItemFeatureBuilder
from app.features.interactions import InteractionMatrix
from app.models.base import BaseRecommender, FitContext, ScoredItem, rank_scores

log = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    name = "content"

    def __init__(
        self,
        half_life_days: float = 30.0,
        min_history: int = 1,
        quality_weight: float = 0.05,
        profile_window: int = 10,
        cache_size: int = 20_000,
        aggregate: str = "mean_max",
        mean_weight: float = 0.5,
        max_weight: float = 0.5,
        **params,
    ) -> None:
        if aggregate not in {"mean", "max", "mean_max", "quality_only"}:
            raise ValueError(
                f"unknown aggregate {aggregate!r}; expected mean, max, mean_max or quality_only"
            )
        super().__init__(
            half_life_days=half_life_days,
            min_history=min_history,
            quality_weight=quality_weight,
            profile_window=profile_window,
            cache_size=cache_size,
            aggregate=aggregate,
            mean_weight=mean_weight,
            max_weight=max_weight,
            **params,
        )
        self.features: ItemFeatureBuilder | None = None
        self._item_matrix: sparse.csr_matrix | None = None
        self._quality: np.ndarray | None = None
        self._interactions: InteractionMatrix | None = None
        self._profile_cache: dict[str, np.ndarray] = {}
        self._max_cache: dict[str, np.ndarray | None] = {}

    # -- fitting ---------------------------------------------------------
    def fit(self, ctx: FitContext) -> ContentBasedRecommender:
        if ctx.features is not None:
            self.features = ctx.features
            self._item_matrix = (
                ctx.features.transform_subset(ctx.product_ids)
                if ctx.item_matrix is None
                else ctx.item_matrix
            )
        else:
            self.features = ItemFeatureBuilder().fit(ctx.products)
            self._item_matrix = self.features.transform_subset(ctx.product_ids)

        ctx.item_matrix = self._item_matrix
        self._finalise(ctx)
        self._quality = self._build_quality(ctx.products)
        self._interactions = ctx.interactions
        self._profile_cache.clear()
        self._max_cache.clear()
        return self

    def _build_quality(self, products: pd.DataFrame) -> np.ndarray:
        """Bayesian average rating x log popularity, as a small cold-start prior."""
        by_id = products.set_index(products["product_id"].astype(str))
        aligned = by_id.reindex(self._item_ids)
        rating = pd.to_numeric(aligned.get("rating"), errors="coerce").fillna(3.5).to_numpy()
        counts = (
            pd.to_numeric(aligned.get("rating_count"), errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        mean_rating, prior = 4.0, 30.0
        bayes = (rating * counts + mean_rating * prior) / (counts + prior)
        pop = np.log1p(counts)
        pop = (pop - pop.min()) / (np.ptp(pop) + 1e-9)
        return 0.7 * ((bayes - 1.0) / 4.0) + 0.3 * pop

    # -- user profile ----------------------------------------------------
    def user_profile(self, user_id: str) -> np.ndarray | None:
        """Recency-weighted centroid of the user's strongest recent items.

        Averaging a user's *entire* history into one vector flattens it: a user
        with 80 events across 6 categories ends up with a profile close to the
        catalogue mean, which ranks everything roughly equally. Restricting the
        centroid to the most recent / highest-weight items keeps the profile
        sharp and reflects the fact that taste drifts over time.
        """
        self._check()
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]
        if self.features is None or self._item_matrix is None:
            return None

        seen = self._interactions.seen_items(user_id)
        if len(seen) < self.params["min_history"]:
            profile = None
        else:
            weights = self._interactions.user_row(user_id).toarray().ravel()[seen]
            window = int(self.params["profile_window"])
            if window > 0 and seen.size > window:
                strongest = np.argsort(-weights)[:window]
                seen, weights = seen[strongest], weights[strongest]
            weights = weights.astype(np.float64)
            # Weighted mean of the history items' feature rows -> a profile in
            # *feature* space, so it can be dotted against the whole catalogue.
            profile = np.asarray(weights @ self._item_matrix[seen, :]).ravel()
            norm = np.linalg.norm(profile)
            profile = profile / norm if norm > 0 else None
        self._cache_profile(user_id, profile)
        return profile

    def invalidate(self, user_id: str | None = None) -> None:
        if user_id is None:
            self._profile_cache.clear()
            self._max_cache.clear()
        else:
            self._profile_cache.pop(user_id, None)
            self._max_cache.pop(user_id, None)

    def _cache_profile(self, user_id: str, profile: np.ndarray | None) -> None:
        """Bounded cache. A long-running service sees unbounded user ids, and
        each entry holds an n_features float vector."""
        if len(self._profile_cache) >= self.params["cache_size"]:
            self._profile_cache.clear()
            self._max_cache.clear()
        self._profile_cache[user_id] = profile

    def update_user(self, user_id: str, product_id: str, weight: float) -> None:
        """Fold a single new event into a cached profile (real-time path).

        The profile is an order-free weighted sum, so absorbing a new item is a
        single add - no retrain, and the next request already reflects the event.
        The stale interaction matrix is *not* rewritten here; that is what the
        periodic retrain is for.
        """
        self._check()
        if self.features is None or self._item_matrix is None:
            return
        if product_id not in self.features.index_by_id:
            log.debug("ignoring update for unknown product %s", product_id)
            return
        vector = np.asarray(self.features.vector_for(product_id).todense()).ravel()
        profile = self._profile_cache.get(user_id)
        combined = (weight * vector) if profile is None else profile + weight * vector
        norm = np.linalg.norm(combined)
        self._cache_profile(user_id, combined / norm if norm > 0 else None)

    # -- scoring ---------------------------------------------------------
    def score(self, user_id: str) -> np.ndarray:
        self._check()
        assert self._item_matrix is not None
        scores = np.zeros(len(self._item_ids), dtype=np.float64)

        if self.params["aggregate"] != "quality_only":
            profile = self.user_profile(user_id)
            if profile is not None:
                scores += self.params["mean_weight"] * np.asarray(
                    self._item_matrix @ profile
                ).ravel()

        max_term = self.params["max_weight"]
        if max_term > 0:
            pooled = self._max_similarity(user_id)
            if pooled is not None:
                scores += max_term * pooled

        if self._quality is not None:
            scores += self.params["quality_weight"] * self._quality
        return scores

    def _max_similarity(self, user_id: str) -> np.ndarray | None:
        """Best match against any single history item.

        A mean-pooled profile can only answer "which category does this user
        like?". Max-pooling additionally answers "which *specific* thing does
        this user like?", which is what separates two products inside the same
        subcategory - and that is exactly where mean-pooling loses to
        collaborative filtering.
        """
        if user_id in self._max_cache:
            return self._max_cache[user_id]
        assert self._item_matrix is not None
        seen = self._interactions.seen_items(user_id)
        if seen.size == 0:
            self._max_cache[user_id] = None
            return None
        # (n_items x n_features) @ (n_features x n_hist) -> similarities to each
        # history item; take the best per catalogue row.
        sims = np.asarray((self._item_matrix @ self._item_matrix[seen].T).todense())
        pooled = sims.max(axis=1)
        if len(self._max_cache) >= self.params["cache_size"]:
            self._max_cache.clear()
        self._max_cache[user_id] = pooled
        return pooled

    def score_many(self, user_ids: list[str]) -> np.ndarray:
        """Vectorised scoring. Identical results to looping over `score`."""
        self._check()
        out = np.zeros((len(user_ids), len(self._item_ids)), dtype=np.float64)
        for row, uid in enumerate(user_ids):
            scores = self.score(uid)
            if scores is not None:
                out[row] = scores
        return out

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude: set[str] | None = None,
        candidates: set[str] | None = None,
        include_seen: bool = False,
    ) -> list[ScoredItem]:
        self._check()
        seen = set(self._interactions_ids(user_id))
        if not include_seen:
            exclude = (exclude or set()) | seen
        return rank_scores(
            self.score(user_id), self._item_ids, top_k,
            exclude=exclude, candidates=candidates, include_seen=include_seen, reason="content",
        )

    def similar(
        self, product_id: str, top_k: int = 10, exclude_self: bool = True
    ) -> list[ScoredItem]:
        """Item-to-item similarity, the 'more like this' endpoint."""
        self._check()
        assert self.features is not None and self._item_matrix is not None
        if product_id not in self.features.index_by_id:
            raise KeyError(product_id)
        vector = self.features.vector_for(product_id)
        sims = np.asarray(self._item_matrix @ vector.T.todense()).ravel()
        exclude = {product_id} if exclude_self else None
        return rank_scores(sims, self._item_ids, top_k, exclude=exclude, reason="similar")

    # -- helpers ---------------------------------------------------------
    def _interactions_ids(self, user_id: str) -> list[str]:
        return self._interactions.item_ids_for_cols(self._interactions.seen_items(user_id))
