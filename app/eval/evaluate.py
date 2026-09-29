"""Offline evaluation harness.

Uses a **temporal** leave-last-N-out split, not a random one. Random splits leak
the future into training and inflate every metric; holding out each user's most
recent positive interactions is the only split that answers "will this work in
production tomorrow?".

Ground truth is restricted to *positive* events (purchase / cart / wishlist). A
`view` is not a prediction that the model got right, so counting views as hits
would reward models for surfacing items users merely glanced at.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.data.synthetic import generate_dataset
from app.domain import POSITIVE_EVENTS
from app.eval import metrics as M
from app.features.build import ItemFeatureBuilder
from app.features.interactions import InteractionMatrix, build_interaction_matrix
from app.models.base import BaseRecommender, FitContext
from app.models.collaborative import CollaborativeRecommender, PopularityRecommender
from app.models.content_based import ContentBasedRecommender
from app.models.hybrid import HybridRecommender

MODELS: dict[str, Callable[[], BaseRecommender]] = {
    "popularity": PopularityRecommender,
    "content": ContentBasedRecommender,
    "collaborative": CollaborativeRecommender,
    "hybrid": HybridRecommender,
}


@dataclass
class TemporalSplit:
    train_events: pd.DataFrame
    test_events: pd.DataFrame
    n_users: int
    cutoff: str
    train_counts: dict[str, int] = field(default_factory=dict)
    val_events: pd.DataFrame = field(default_factory=pd.DataFrame)

    @staticmethod
    def _truth(frame: pd.DataFrame) -> dict[str, set[str]]:
        if frame.empty:
            return {}
        return (
            frame.groupby("user_id")["product_id"]
            .apply(lambda s: set(s.astype(str)))
            .to_dict()
        )

    @property
    def test_truth(self) -> dict[str, set[str]]:
        return self._truth(self.test_events)

    @property
    def val_truth(self) -> dict[str, set[str]]:
        return self._truth(self.val_events)

    def truth_for(self, target: str) -> dict[str, set[str]]:
        return self.val_truth if target == "val" else self.test_truth

    def users_with_history(self, low: int | None = None, high: int | None = None) -> list[str]:
        """Users whose training history falls in [low, high].

        Slicing by history length is what separates the cold-start question
        ("can we recommend to someone we know almost nothing about?") from the
        warm question. Averaging the two together hides exactly the regime
        where content-based earns its keep.
        """
        counts = self.train_counts
        return sorted(
            uid
            for uid in counts
            if (low is None or counts[uid] >= low) and (high is None or counts[uid] <= high)
        )


COLD_START_MAX_EVENTS = 25
WARM_MIN_EVENTS = 60


@dataclass
class EvalResult:
    model: str
    k: int
    n_users: int
    metrics: dict[str, float]
    segment: str = "all"
    fit_seconds: float = 0.0
    score_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def temporal_split(
    events: pd.DataFrame,
    holdout_per_user: int = 5,
    val_size: int = 2,
    positive_only: bool = True,
    min_train_events: int = 5,
) -> TemporalSplit:
    """Time-ordered train / validation / test split.

    Each user's last ``holdout_per_user`` positive events are held out and then
    cut in chronological half: the earlier slice becomes *validation* and the
    later slice becomes *test*.

    Training is additionally cut off at each user's *own* first held-out event,
    not just at the held-out rows. Without that, a user's trailing `view`
    events - which happen after their last purchase - would stay in the
    training set, and the model would be scored on a period it had already seen
    the tail of. That is real temporal leakage, and it flatters every model.
    """
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    if positive_only:
        candidates = frame[frame["event_type"].isin(POSITIVE_EVENTS)]
    else:
        candidates = frame

    # Rank each user's events newest-first and take the top N.
    candidates = candidates.sort_values(["user_id", "timestamp"], ascending=[True, False])
    holdout = candidates.groupby("user_id", group_keys=False).head(holdout_per_user)

    # Per-user cutoff: the earliest held-out event.
    cutoffs = holdout.groupby("user_id")["timestamp"].min()
    train = frame[
        ~frame["user_id"].isin(cutoffs.index)
        | (frame["timestamp"] < frame["user_id"].map(cutoffs))
    ]

    # Users need enough history to build a profile.
    train_counts = train.groupby("user_id").size()
    eligible = set(train_counts[train_counts >= min_train_events].index)
    train = train[train["user_id"].isin(eligible)]
    holdout = holdout[holdout["user_id"].isin(eligible)]

    # Cut the holdout into val (the older slice) and test (the newer slice)
    # within each user, so validation sits strictly before the scoring window.
    holdout = holdout.sort_values(["user_id", "timestamp"], ascending=[True, True])
    val = holdout.groupby("user_id", group_keys=False).head(val_size)
    test = holdout[~holdout.index.isin(val.index)]

    # Drop any held-out item the user already interacted with before the cutoff:
    # we cannot expect a recommender to re-recommend something it has seen.
    seen = set(zip(train["user_id"], train["product_id"], strict=True))
    val = val[[(u, p) not in seen for u, p in zip(val["user_id"], val["product_id"], strict=True)]]
    test = test[
        [(u, p) not in seen for u, p in zip(test["user_id"], test["product_id"], strict=True)]
    ]

    cutoff = frame["timestamp"].min().isoformat() if not frame.empty else ""
    counts = train.groupby("user_id").size().to_dict()
    return TemporalSplit(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        len(eligible),
        cutoff,
        {str(k): int(v) for k, v in counts.items()},
        val.reset_index(drop=True),
    )


def build_models(settings: Settings | None = None) -> dict[str, BaseRecommender]:
    settings = settings or get_settings()
    content = ContentBasedRecommender(half_life_days=settings.recency_halflife_days)
    collaborative = CollaborativeRecommender(
        knn_top_k=settings.knn_top_k, n_factors=settings.svd_factors
    )
    return {
        "popularity": PopularityRecommender(),
        "content": content,
        "collaborative": collaborative,
        "hybrid": HybridRecommender(content=content, collaborative=collaborative),
    }


def evaluate_model(
    model: BaseRecommender,
    split: TemporalSplit,
    products: pd.DataFrame,
    k_values: tuple[int, ...],
    features: ItemFeatureBuilder | None = None,
    half_life_days: float = 30.0,
    catalog: set[str] | None = None,
    segment: str = "all",
    user_subset: list[str] | None = None,
    target: str = "test",
    all_segments: bool = False,
) -> list[EvalResult]:
    """Fit on the training split, then score one or more user segments."""
    segments = _segment_groups(split) if all_segments else {segment: user_subset}
    return evaluate_segments(
        model, split, products, k_values, segments,
        features=features, half_life_days=half_life_days,
        catalog=catalog, target=target,
    )


def evaluate_segments(
    model: BaseRecommender,
    split: TemporalSplit,
    products: pd.DataFrame,
    k_values: tuple[int, ...],
    segments: dict[str, list[str] | None],
    features: ItemFeatureBuilder | None = None,
    half_life_days: float = 30.0,
    catalog: set[str] | None = None,
    target: str = "test",
) -> list[EvalResult]:
    """Fit the model **once**, then score any number of user segments.

    The segments differ only in who gets scored, never in what the model was
    trained on, so refitting per segment would just multiply the cost.
    """
    product_ids = products["product_id"].astype(str).tolist()
    interactions = build_interaction_matrix(
        split.train_events, product_ids, halflife_days=half_life_days
    )
    ctx = FitContext.build(products, interactions, features=features)

    t0 = time.perf_counter()
    model.fit(ctx)
    fit_seconds = time.perf_counter() - t0

    if features is not None and model.fitted:
        item_matrix = features.transform_subset(product_ids)
    else:
        item_matrix = getattr(model, "_item_matrix", None)

    results: list[EvalResult] = []
    for name, subset in segments.items():
        if name != "all" and not subset:
            continue
        results.extend(
            _score_segment(
                model, split, interactions, item_matrix, features, target,
                k_values, name, subset, fit_seconds,
                half_life_days=half_life_days, catalog=catalog,
            )
        )
    return results


def _segment_groups(split: TemporalSplit) -> dict[str, list[str] | None]:
    return {
        "all": None,
        "cold_start": split.users_with_history(high=COLD_START_MAX_EVENTS),
        "warm": split.users_with_history(low=WARM_MIN_EVENTS),
    }


def _score_segment(
    model: BaseRecommender,
    split: TemporalSplit,
    interactions: InteractionMatrix,
    item_matrix,
    features: ItemFeatureBuilder | None,
    target: str,
    k_values: tuple[int, ...],
    segment: str,
    user_subset: list[str] | None,
    fit_seconds: float,
    half_life_days: float = 30.0,
    catalog: set[str] | None = None,
) -> list[EvalResult]:
    """Rank for one user segment and reduce to per-k metrics.

    Split out from `evaluate_model` so a caller that wants several segments
    pays for one fit rather than one fit per segment.
    """
    truth = split.truth_for(target)
    user_ids = list(user_subset) if user_subset is not None else sorted(truth)
    # A user can be in the segment but have no held-out positives left after the
    # "already seen in training" filter; they contribute nothing to the metrics.
    user_ids = [u for u in user_ids if truth.get(u)]
    product_ids = interactions.item_ids

    similarity = item_matrix
    item_index = features.index_by_id if features is not None else None
    popularity = _popularity_dict(interactions)

    # Score once at the widest k, then derive every smaller k as a prefix.
    max_k = max(k_values)
    t0 = time.perf_counter()
    recommendations: dict[str, list[str]] = {}
    train_seen = _seen_sets(interactions)
    for uid in user_ids:
        if not train_seen.get(uid):
            continue
        ranked = model.recommend(uid, top_k=max_k)
        recommendations[uid] = [item.product_id for item in ranked]
    score_seconds = time.perf_counter() - t0

    catalog = catalog or set(product_ids)
    results: list[EvalResult] = []
    for k in k_values:
        per_user: dict[str, list[float]] = {
            "precision": [], "recall": [], "map": [], "ndcg": [], "mrr": [], "hit_rate": [],
        }
        all_lists: list[list[str]] = []
        for uid, recs in recommendations.items():
            relevant = truth.get(uid)
            if not relevant:
                continue
            all_lists.append(recs[:k])
            per_user["precision"].append(M.precision_at_k(recs, relevant, k))
            per_user["recall"].append(M.recall_at_k(recs, relevant, k))
            per_user["map"].append(M.average_precision_at_k(recs, relevant, k))
            per_user["ndcg"].append(M.ndcg_at_k(recs, relevant, k))
            per_user["mrr"].append(M.reciprocal_rank_at_k(recs, relevant, k))
            per_user["hit_rate"].append(M.hit_rate_at_k(recs, relevant, k))

        summary = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in per_user.items()}
        summary["coverage"] = M.coverage(all_lists, catalog)
        summary["novelty"] = float(
            np.mean([M.novelty(r, popularity) for r in all_lists]) if all_lists else 0.0
        )
        summary["personalisation"] = M.personalisation(all_lists)
        summary["diversity"] = float(
            np.mean(
                [M.intra_list_diversity(r, similarity, item_index) for r in all_lists]
            )
            if all_lists and similarity is not None
            else 0.0
        )
        summary["mean_recommendations"] = (
            float(np.mean([len(r) for r in all_lists])) if all_lists else 0.0
        )

        results.append(
            EvalResult(
                model=model.name,
                k=k,
                n_users=len(all_lists),
                segment=segment,
                metrics=summary,
                fit_seconds=fit_seconds,
                score_seconds=score_seconds,
                extra={"catalog_size": len(catalog)},
            )
        )
    return results


def _seen_sets(interactions: InteractionMatrix) -> dict[str, set[str]]:
    matrix = interactions.matrix.tocsr()
    out: dict[str, set[str]] = {}
    for i, uid in enumerate(interactions.user_ids):
        cols = matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]]
        if cols.size:
            out[uid] = set(interactions.item_ids_for_cols(cols))
    return out


def _popularity_dict(interactions: InteractionMatrix) -> dict[str, float]:
    totals = np.asarray(interactions.matrix.sum(axis=0)).ravel()
    total = totals.sum()
    if total <= 0:
        return {p: 0.0 for p in interactions.item_ids}
    return {
        p: float(t / total) for p, t in zip(interactions.item_ids, totals, strict=True)
    }


def evaluate_all(
    events: pd.DataFrame | None = None,
    products: pd.DataFrame | None = None,
    settings: Settings | None = None,
    model_names: list[str] | None = None,
    segments: list[str] | None = None,
) -> tuple[list[EvalResult], TemporalSplit]:
    """Full offline comparison across every model strategy and user segment.

    Each model is fitted once per segment group; the segments differ only in
    *which users are scored*, never in what the model was trained on. Comparing
    them is the only way to see that content-based and collaborative filtering
    fail in different regimes rather than one being universally better.
    """
    settings = settings or get_settings()
    if events is None or products is None:
        generated = generate_dataset(settings)
        events = events if events is not None else generated["events"]
        products = products if products is not None else generated["products"]

    split = temporal_split(events, holdout_per_user=settings.eval_holdout_per_user)
    features = ItemFeatureBuilder().fit(products)
    catalog = set(products["product_id"].astype(str))

    segments = segments or ["all", "cold_start", "warm"]
    keep = {s for s in segments if s != "all"}
    results: list[EvalResult] = []
    for name, model in build_models(settings).items():
        if model_names and name not in model_names:
            continue
        for result in evaluate_model(
            model, split, products, settings.eval_top_k,
            features=features, half_life_days=settings.recency_halflife_days,
            catalog=catalog, target="test", all_segments=True,
        ):
            if result.segment == "all" or result.segment in keep:
                results.append(result)
    return results, split


def results_to_frame(results: list[EvalResult]) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        row = {
            "segment": r.segment,
            "model": r.model,
            "k": r.k,
            "n_users": r.n_users,
            "fit_s": round(r.fit_seconds, 3),
            "score_s": round(r.score_seconds, 3),
        }
        row.update({m: round(v, 5) for m, v in r.metrics.items()})
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["segment", "k", "ndcg"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
