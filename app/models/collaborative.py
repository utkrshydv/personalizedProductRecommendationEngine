"""Collaborative filtering.

Two complementary estimators are trained on the same implicit-feedback matrix
and blended:

* **Item-item kNN** - cosine similarity between the columns of the
  user x item matrix, i.e. "users who bought X also bought Y". Scores for a
  user are the similarity-weighted sum of that user's history. It is sharp on
  head items and needs no training loop.
* **Truncated SVD** - a rank-f factorisation of the user x item matrix,
  scoring by the user factor's dot product with every item factor. It
  generalises across sparse, weakly-connected items where kNN has no
  co-occurrence evidence at all.

The blend is `knn_weight * knn + (1 - knn_weight) * svd`, with both components
min-max normalised per user so the weights are comparable.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from app.features.interactions import InteractionMatrix
from app.models.base import BaseRecommender, FitContext, ScoredItem, rank_scores


class ItemItemKNN:
    """Cosine item-item similarity over the user x item interaction matrix."""

    def __init__(self, top_k: int = 100, shrink: float = 0.0) -> None:
        self.top_k = top_k
        self.shrink = shrink
        self.similarity_: sparse.csr_matrix | None = None
        #: Dense (unpruned) similarity. Kept so symmetry can be asserted after
        #: shrinkage; `similarity_` is deliberately asymmetric because keeping
        #: only each row's top-k drops (i, j) while keeping (j, i).
        self.similarity_full_: sparse.csr_matrix | None = None
        self.popularity_: np.ndarray | None = None

    def fit(self, matrix: sparse.csr_matrix) -> ItemItemKNN:
        matrix = matrix.tocsr().astype(np.float32)
        # Item popularity, used to damp similarity for rarely-co-consumed items.
        counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
        self.popularity_ = counts

        unit = normalize(matrix, norm="l2", axis=1, copy=True)
        sims = (unit.T @ unit).tocsr()

        if self.shrink > 0:
            # Shrinkage (Bell & Koren): damp a pair by how little evidence
            # supports it. The denominator must be symmetric in (i, j), both to
            # keep S_ij == S_ji - "users who bought X also bought Y" is a
            # symmetric claim - and because scoring multiplies a user's history
            # by S, where an asymmetric S quietly biases towards popular items.
            coo = sims.tocoo()
            evidence = counts[coo.row] + counts[coo.col]
            coo.data = coo.data / (1.0 + evidence / self.shrink)
            sims = coo.tocsr()

        sims.setdiag(0.0)
        sims.eliminate_zeros()
        self.similarity_full_ = sims.copy()
        self.similarity_ = _top_k_per_row(sims, self.top_k)
        return self

    def transform(self, matrix: sparse.csr_matrix) -> np.ndarray:
        """Score matrix for every user: R @ S."""
        if self.similarity_ is None:
            raise RuntimeError("ItemItemKNN must be fitted first")
        return np.asarray((matrix @ self.similarity_).todense(), dtype=np.float32)


class LatentFactorModel:
    """TruncatedSVD on the implicit matrix, exposing user and item factors."""

    def __init__(self, n_factors: int = 64, n_iter: int = 15, random_state: int = 42) -> None:
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_: TruncatedSVD | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None

    def fit(self, matrix: sparse.csr_matrix) -> LatentFactorModel:
        n_components = int(min(self.n_factors, max(2, min(matrix.shape) - 1)))
        self.model_ = TruncatedSVD(
            n_components=n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.user_factors_ = self.model_.fit_transform(matrix).astype(np.float32)
        self.item_factors_ = self.model_.components_.T.astype(np.float32)
        return self

    def transform(self, matrix: sparse.csr_matrix) -> np.ndarray:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("LatentFactorModel must be fitted first")
        user_factors = (
            self.user_factors_
            if matrix.shape[0] == self.user_factors_.shape[0]
            else self.model_.transform(matrix).astype(np.float32)
        )
        return user_factors @ self.item_factors_.T

    def score_rows(self, rows: np.ndarray) -> np.ndarray:
        """Scores for specific user rows, reusing the cached factors.

        `transform` re-projects when the row count does not match, and one
        randomised SVD projection per user turns a 300-user evaluation into
        minutes of work. The factors are already computed at fit time, so a
        known user row is just a slice and a matmul.
        """
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("LatentFactorModel must be fitted first")
        rows = np.asarray(rows, dtype=int)
        if rows.size and rows.max() < self.user_factors_.shape[0]:
            return self.user_factors_[rows] @ self.item_factors_.T
        return self.model_.transform(sparse.csr_matrix((len(rows), self.item_factors_.shape[0]))) \
            .astype(np.float32) @ self.item_factors_.T

    @property
    def explained_variance(self) -> float:
        return float(self.model_.explained_variance_ratio_.sum()) if self.model_ else 0.0


def _top_k_per_row(matrix: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    """Keep only the top_k entries of every row (cosine-similarity pruning)."""
    matrix = matrix.tocsr()
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    keep_data: list[np.ndarray] = []
    keep_idx: list[np.ndarray] = []
    new_indptr = [0]
    for i in range(matrix.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        row_data = data[start:end]
        row_idx = indices[start:end]
        if row_data.size > top_k:
            keep = np.argpartition(-row_data, kth=top_k - 1)[:top_k]
            row_data, row_idx = row_data[keep], row_idx[keep]
        keep_data.append(row_data)
        keep_idx.append(row_idx)
        new_indptr.append(new_indptr[-1] + row_data.size)
    return sparse.csr_matrix(
        (
            np.concatenate(keep_data) if keep_data else np.array([], dtype=matrix.dtype),
            np.concatenate(keep_idx) if keep_idx else np.array([], dtype=indices.dtype),
            np.array(new_indptr, dtype=indptr.dtype),
        ),
        shape=matrix.shape,
    )


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise min-max to [0, 1]; all-equal rows map to zeros."""
    if scores.size == 0:
        return scores
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    spread = hi - lo
    out = np.zeros_like(scores, dtype=np.float32)
    np.divide(scores - lo, spread, out=out, where=spread > 1e-12)
    return np.nan_to_num(out)


class CollaborativeRecommender(BaseRecommender):
    """kNN + latent-factor blend, exposed as one recommender."""

    name = "collaborative"

    def __init__(
        self,
        knn_top_k: int = 100,
        n_factors: int = 64,
        knn_weight: float = 0.5,
        shrink: float = 10.0,
        n_iter: int = 15,
        random_state: int = 42,
        **params,
    ) -> None:
        super().__init__(
            knn_top_k=knn_top_k,
            n_factors=n_factors,
            knn_weight=knn_weight,
            shrink=shrink,
            n_iter=n_iter,
            random_state=random_state,
            **params,
        )
        self.knn: ItemItemKNN | None = None
        self.latent: LatentFactorModel | None = None
        self._interactions: InteractionMatrix | None = None
        self._cache: dict[int, np.ndarray] = {}
        self._popularity: np.ndarray | None = None

    # -- fitting ---------------------------------------------------------
    def fit(self, ctx: FitContext) -> CollaborativeRecommender:
        matrix = ctx.interactions.matrix
        self._interactions = ctx.interactions
        self.knn = ItemItemKNN(
            top_k=self.params["knn_top_k"], shrink=self.params["shrink"]
        ).fit(matrix)
        self.latent = LatentFactorModel(
            n_factors=self.params["n_factors"],
            n_iter=self.params["n_iter"],
            random_state=self.params["random_state"],
        ).fit(matrix)
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        self._popularity = np.log1p(counts)
        total = self._popularity.sum()
        self._popularity = self._popularity / total if total > 0 else self._popularity
        self._cache.clear()
        self._finalise(ctx)
        return self

    # -- scoring ---------------------------------------------------------
    def _blend(self, rows: np.ndarray) -> np.ndarray:
        assert self.knn is not None and self.latent is not None
        matrix = self._interactions.matrix[rows]
        knn_scores = minmax_rows(self.knn.transform(matrix))
        svd_scores = minmax_rows(self.latent.score_rows(rows))
        w = self.params["knn_weight"]
        return (w * knn_scores + (1.0 - w) * svd_scores).astype(np.float32)

    def knn_scores_for(self, rows: np.ndarray) -> np.ndarray:
        """Item-kNN component only. Exposed for ablation and for blending."""
        assert self.knn is not None
        return minmax_rows(self.knn.transform(self._interactions.matrix[np.asarray(rows, int)]))

    def svd_scores_for(self, rows: np.ndarray) -> np.ndarray:
        """Latent-factor component only."""
        assert self.latent is not None
        return minmax_rows(self.latent.score_rows(rows))

    def score(self, user_id: str) -> np.ndarray:
        self._check()
        row = self._interactions.user_index(user_id)
        if row is None:
            return np.zeros(len(self._item_ids), dtype=np.float32)
        return self._blend(np.array([row]))[0]

    def score_batch(self, user_ids: list[str]) -> np.ndarray:
        """Vectorised scoring for offline evaluation."""
        self._check()
        rows = [self._interactions.user_index(u) for u in user_ids]
        known = [i for i, r in enumerate(rows) if r is not None]
        out = np.zeros((len(user_ids), len(self._item_ids)), dtype=np.float32)
        if known:
            out[known] = self._blend(np.array([rows[i] for i in known], dtype=int))
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
        seen = set(
            self._interactions.item_ids_for_cols(self._interactions.seen_items(user_id))
        )
        if not include_seen:
            exclude = (exclude or set()) | seen
        return rank_scores(
            self.score(user_id), self._item_ids, top_k,
            exclude=exclude, candidates=candidates,
            include_seen=include_seen, reason="collaborative",
        )

    def similar(
        self, product_id: str, top_k: int = 10, exclude_self: bool = True
    ) -> list[ScoredItem]:
        self._check()
        assert self.knn is not None
        col = self._interactions.item_index(product_id)
        if col is None:
            raise KeyError(product_id)
        sims = np.asarray(self.knn.similarity_.getcol(col).todense()).ravel()
        exclude = {product_id} if exclude_self else None
        return rank_scores(sims, self._item_ids, top_k, exclude=exclude, reason="similar")

    def explain(self) -> dict:
        return {
            "explained_variance": self.latent.explained_variance if self.latent else 0.0,
            "n_factors": int(self.latent.user_factors_.shape[1]) if self.latent else 0,
            "knn_nnz": int(self.knn.similarity_.nnz) if self.knn else 0,
        }


class PopularityRecommender(BaseRecommender):
    """Non-personalised baseline. The bar any real model has to clear."""

    name = "popularity"

    def __init__(self, recency_half_life_days: float = 30.0, **params) -> None:
        super().__init__(recency_half_life_days=recency_half_life_days, **params)
        self._scores: np.ndarray | None = None
        self._interactions: InteractionMatrix | None = None

    def fit(self, ctx: FitContext) -> PopularityRecommender:
        matrix = ctx.interactions.matrix
        totals = np.asarray(matrix.sum(axis=0)).ravel()
        self._scores = np.log1p(totals).astype(np.float32)
        self._interactions = ctx.interactions
        self._finalise(ctx)
        return self

    def score(self, user_id: str) -> np.ndarray:
        self._check()
        return self._scores

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude: set[str] | None = None,
        candidates: set[str] | None = None,
        include_seen: bool = False,
    ) -> list[ScoredItem]:
        self._check()
        if not include_seen:
            exclude = (exclude or set()) | set(
                self._interactions.item_ids_for_cols(self._interactions.seen_items(user_id))
            )
        return rank_scores(
            self._scores, self._item_ids, top_k,
            exclude=exclude, candidates=candidates, include_seen=include_seen, reason="popularity",
        )

    def recommend_global(self, top_k: int = 10) -> list[ScoredItem]:
        """Catalogue-level trending list, independent of any user."""
        self._check()
        return rank_scores(self._scores, self._item_ids, top_k, reason="trending")
