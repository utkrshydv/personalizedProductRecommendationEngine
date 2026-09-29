"""Hyperparameter search over the fusion weights.

Fusion weights are the one set of parameters that directly trades accuracy
against coverage and personalisation, so they are worth choosing properly
rather than guessing. They are tuned on the *validation* slice of the temporal
split - which sits strictly between train and test - and the test slice is only
ever touched to produce the final report.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.eval.evaluate import (
    COLD_START_MAX_EVENTS,
    TemporalSplit,
    build_models,
    evaluate_segments,
    temporal_split,
)
from app.features.build import ItemFeatureBuilder
from app.models.base import BaseRecommender
from app.models.collaborative import CollaborativeRecommender
from app.models.hybrid import HybridRecommender

log = logging.getLogger(__name__)

#: Candidate fusion weights. The three are normalised at scoring time, so only
#: their ratio matters - the grid is over ratios.
CONTENT_WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)
COLLAB_WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
KNN_WEIGHTS = (0.3, 0.5, 0.7)

#: Segments averaged into the tuning objective. Cold-start and warm users value
#: accuracy very differently, so tuning on the pooled average would just optimise
#: for the warm majority.
TUNING_SEGMENTS = ("all", "cold_start")


@dataclass
class TuningResult:
    best_params: dict = field(default_factory=dict)
    baseline_ndcg: float = 0.0
    tuned_ndcg: float = 0.0
    baseline_coverage: float = 0.0
    tuned_coverage: float = 0.0
    coverage_weight: float = 0.0
    trials: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "objective": (
                f"mean NDCG@10 over {list(TUNING_SEGMENTS)} "
                f"+ {self.coverage_weight} * mean coverage"
            ),
            "validation_ndcg": self.tuned_ndcg,
            "validation_coverage": self.tuned_coverage,
            "collaborative_only_ndcg": self.baseline_ndcg,
            "collaborative_only_coverage": self.baseline_coverage,
            "ndcg_lift_over_collaborative": (
                round(self.tuned_ndcg / self.baseline_ndcg - 1.0, 4)
                if self.baseline_ndcg
                else 0.0
            ),
            "coverage_lift_over_collaborative": (
                round(self.tuned_coverage - self.baseline_coverage, 4)
                if self.baseline_coverage
                else 0.0
            ),
            "n_trials": len(self.trials),
        }


def _build_candidate(
    content_weight: float,
    collab_weight: float,
    knn_weight: float,
    settings: Settings,
) -> BaseRecommender:
    collaborative = CollaborativeRecommender(
        knn_top_k=settings.knn_top_k,
        n_factors=settings.svd_factors,
        knn_weight=knn_weight,
    )
    content = build_models(settings)["content"]
    return HybridRecommender(
        content=content,
        collaborative=collaborative,
        content_weight=content_weight,
        collab_weight=collab_weight,
        popularity_weight=max(0.0, 1.0 - content_weight - collab_weight),
    )


def _objective(results: list, weights: dict[str, float], coverage_weight: float = 0.0) -> float:
    """Segment-weighted NDCG@10, plus a coverage bonus.

    Accuracy alone is the wrong single number to tune against. Content-based
    ranking contributes comparatively little NDCG but a great deal of catalogue
    reach (coverage ~0.90 vs ~0.65 for collaborative filtering), and reach is
    what keeps a catalogue sellable rather than collapsing onto the same head
    items for everyone. `coverage_weight` makes that trade explicit and
    tunable; set it to 0 for a pure-accuracy objective.
    """
    by_segment: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r.k != 10:
            continue
        bucket = by_segment.setdefault(r.segment, {"ndcg": [], "coverage": []})
        bucket["ndcg"].append(r.metrics["ndcg"])
        bucket["coverage"].append(r.metrics.get("coverage", 0.0))

    score = 0.0
    total = 0.0
    for segment, weight in weights.items():
        bucket = by_segment.get(segment)
        if not bucket:
            continue
        ndcg = float(np.mean(bucket["ndcg"]))
        coverage = float(np.mean(bucket["coverage"]))
        score += weight * (ndcg + coverage_weight * coverage)
        total += weight
    return score / total if total else 0.0


def tune_fusion_weights(
    events: pd.DataFrame,
    products: pd.DataFrame,
    settings: Settings | None = None,
    verbose: bool = False,
) -> tuple[TuningResult, TemporalSplit]:
    """Grid-search hybrid fusion weights against the validation slice."""
    settings = settings or get_settings()
    split = temporal_split(
        events,
        holdout_per_user=settings.eval_holdout_per_user + settings.tuning_val_size,
        val_size=settings.tuning_val_size,
    )
    log.info(
        "tuning split: train=%d val=%d test=%d",
        len(split.train_events), len(split.val_events), len(split.test_events),
    )

    features = ItemFeatureBuilder().fit(products)
    catalog = set(products["product_id"].astype(str))
    groups = {
        "all": None,
        "cold_start": split.users_with_history(high=COLD_START_MAX_EVENTS),
    }
    objective_weights = {s: 1.0 for s in TUNING_SEGMENTS}
    cov_w = settings.tuning_coverage_weight

    # Score a sample of users. The relative ordering of ~300 candidate weight
    # vectors is stable, and it keeps the search interactive.
    sample_rng = np.random.default_rng(settings.tuning_seed)
    val_users = np.array(sorted(split.val_truth))
    if len(val_users) > settings.tuning_max_users:
        val_users = np.sort(
            sample_rng.choice(val_users, size=settings.tuning_max_users, replace=False)
        )
    for segment, subset in list(groups.items()):
        if subset is None:
            groups[segment] = list(val_users)
        else:
            picked = sorted(set(subset) & set(val_users.tolist()))
            if not picked:
                groups.pop(segment)

    # Baseline: collaborative only, i.e. the fusion that ignores content.
    baseline = evaluate_segments(
        CollaborativeRecommender(
            knn_top_k=settings.knn_top_k, n_factors=settings.svd_factors
        ),
        split, products, (10,), {"all": groups["all"]},
        features=features, half_life_days=settings.recency_halflife_days,
        catalog=catalog, target="val",
    )
    baseline_ndcg = float(np.mean([r.metrics["ndcg"] for r in baseline if r.k == 10]))
    baseline_coverage = float(
        np.mean([r.metrics["coverage"] for r in baseline if r.k == 10])
    )

    trials: list[dict] = []
    best: tuple[float, dict] | None = None
    grid = [
        (c, k, n)
        for c, k, n in itertools.product(CONTENT_WEIGHTS, COLLAB_WEIGHTS, KNN_WEIGHTS)
        if c + k < 1.0
    ]
    log.info("grid search over %d weight combinations", len(grid))
    total = len(grid)
    for trial_no, (content_w, collab_w, knn_w) in enumerate(grid, start=1):
        model = _build_candidate(content_w, collab_w, knn_w, settings)
        # One fit, both segments: refitting per segment doubled the cost of the
        # search for no benefit, since the training data never changes.
        scored = evaluate_segments(
            model, split, products, (10,), groups,
            features=features, half_life_days=settings.recency_halflife_days,
            catalog=catalog, target="val",
        )
        score = _objective(scored, objective_weights, coverage_weight=cov_w)
        ndcg = float(
            np.mean([r.metrics["ndcg"] for r in scored if r.k == 10 and r.segment == "all"])
        )
        coverage = float(
            np.mean(
                [r.metrics["coverage"] for r in scored if r.k == 10 and r.segment == "all"]
            )
        )
        params = {
            "content_weight": content_w,
            "collab_weight": collab_w,
            "knn_weight": knn_w,
        }
        trials.append(
            {
                **params,
                "objective": round(score, 5),
                "validation_ndcg": round(ndcg, 5),
                "validation_coverage": round(coverage, 4),
            }
        )
        if best is None or score > best[0]:
            best = (score, params)
        if trial_no % 5 == 0 or trial_no == total:
            log.info(
                "[%d/%d] content=%.1f collab=%.1f knn=%.1f -> objective=%.5f "
                "ndcg=%.5f coverage=%.4f (best so far %.5f)",
                trial_no, total, content_w, collab_w, knn_w, score, ndcg, coverage,
                best[0] if best else 0.0,
            )
        if verbose:
            log.debug("trial %s -> objective=%.5f ndcg=%.5f coverage=%.4f",
                      params, score, ndcg, coverage)

    assert best is not None, "no fusion weights were evaluated"
    trials.sort(key=lambda t: -t["objective"])
    winner = trials[0]

    return (
        TuningResult(
            best_params=best[1],
            baseline_ndcg=baseline_ndcg,
            tuned_ndcg=winner["validation_ndcg"],
            baseline_coverage=baseline_coverage,
            tuned_coverage=winner["validation_coverage"],
            coverage_weight=cov_w,
            trials=trials,
        ),
        split,
    )


def save_tuning(result: TuningResult, path: Path | None = None) -> Path:
    settings = get_settings()
    path = path or (settings.artifact_dir / "fusion_weights.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["grid"] = {
        "content_weight": list(CONTENT_WEIGHTS),
        "collab_weight": list(COLLAB_WEIGHTS),
        "knn_weight": list(KNN_WEIGHTS),
        "segments": list(TUNING_SEGMENTS),
    }
    payload["top_trials"] = result.trials[:15]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
