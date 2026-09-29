"""End-to-end training pipeline.

Stages
    1. ingest   - load (or generate) the dataset into the store
    2. features - fit the item feature builder on the catalogue
    3. tune     - grid-search fusion weights on the validation slice
    4. fit      - train every strategy on the full training split
    5. evaluate - score the held-out test slice, per user segment
    6. persist  - write the bundle, metrics and metadata

Every stage is separately callable so the API can retrain a single piece
without a full rebuild.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.data.synthetic import generate_dataset
from app.db.base import Store
from app.domain import Event, Product, User
from app.eval.evaluate import (
    COLD_START_MAX_EVENTS,
    WARM_MIN_EVENTS,
    evaluate_segments,
    results_to_frame,
    temporal_split,
)
from app.eval.tuning import save_tuning, tune_fusion_weights
from app.features.build import ItemFeatureBuilder
from app.features.interactions import build_interaction_matrix
from app.models.base import FitContext
from app.models.collaborative import CollaborativeRecommender
from app.models.content_based import ContentBasedRecommender
from app.models.hybrid import HybridRecommender
from app.training.artifacts import ModelBundle, data_fingerprint, save_bundle, utcnow

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    products: int
    users: int
    events: int
    source: str
    seconds: float


def ingest(
    store: Store,
    settings: Settings | None = None,
    generate: bool = False,
    clear: bool = True,
) -> IngestReport:
    """Populate the store, generating the synthetic dataset if asked."""
    settings = settings or get_settings()
    t0 = time.perf_counter()
    if clear:
        store.clear()

    if generate:
        log.info("generating synthetic dataset (users=%d products=%d events=%d)",
                 settings.n_users, settings.n_products, settings.n_events)
        frames = generate_dataset(settings)
        source = "synthetic"
    else:
        products = store.products()
        users = store.users()
        events = store.events()
        if products.empty or events.empty:
            raise RuntimeError(
                "store is empty; re-run with generate=True to seed it with the "
                "synthetic dataset (or load your own data first)"
            )
        frames = {"products": products, "users": users, "events": events}
        source = f"{store.backend}:existing"

    store.upsert_products(Product.from_dict(r) for r in frames["products"].to_dict("records"))
    if not frames["users"].empty:
        store.upsert_users(User.from_dict(r) for r in frames["users"].to_dict("records"))
    store.insert_events(
        Event.from_dict(r)
        for r in frames["events"].to_dict("records")
        if all(k in r and r[k] is not None for k in
               ("event_id", "user_id", "product_id", "event_type", "timestamp"))
    )

    counts = store.counts()
    report = IngestReport(
        products=counts["products"], users=counts["users"], events=counts["events"],
        source=source, seconds=time.perf_counter() - t0,
    )
    log.info("ingested %s in %.2fs", report, report.seconds)
    return report


def train(
    store: Store,
    settings: Settings | None = None,
    tune_weights: bool = True,
) -> tuple[ModelBundle, list[dict]]:
    """Run the full pipeline and persist the artifacts. Returns the bundle."""
    settings = settings or get_settings()

    products = store.products()
    events = store.events()
    if products.empty or events.empty:
        raise RuntimeError("cannot train on an empty store; call ingest() first")
    products["product_id"] = products["product_id"].astype(str)
    events["product_id"] = events["product_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)

    # ---- stage 2: temporal split + features -------------------------------
    tuning_result = None
    if tune_weights:
        log.info("tuning fusion weights on the validation slice")
        tuning_result, split = tune_fusion_weights(events, products, settings)
        log.info("best fusion weights: %s (val ndcg@10=%.5f)",
                 tuning_result.best_params, tuning_result.tuned_ndcg)
        if tuning_result is not None:
            save_tuning(tuning_result)
    else:
        split = temporal_split(
            events,
            holdout_per_user=settings.eval_holdout_per_user + settings.tuning_val_size,
            val_size=settings.tuning_val_size,
        )

    features = ItemFeatureBuilder().fit(products)
    catalog = set(products["product_id"])

    # ---- stage 3: fit the served models on the full event log ---------------
    # The temporal cut exists so the *evaluation* below is honest. It must not
    # be applied to the model that actually serves traffic: a user whose events
    # all post-date the cutoff would be missing from the served matrix and get
    # cold-start handling despite having a full history in the store.
    log.info("fitting served models on all %d events", len(events))
    interactions = build_interaction_matrix(
        events, list(catalog), halflife_days=settings.recency_halflife_days
    )
    ctx = FitContext.build(products, interactions, features=features)

    params = fusion_params(settings, tuning_result)
    models = {
        name: model.fit(ctx) for name, model in build_models_with(settings, params).items()
    }
    served_events = events

    # ---- stage 4: evaluate on the untouched test slice --------------------
    # The models scored below are rebuilt from the same hyperparameters and
    # fitted by `evaluate_segments` on the *train* split only, so the reported
    # metrics are never measured against a model that has seen the test period.
    results: list = []
    eval_segments = {
        "all": None,
        "cold_start": split.users_with_history(high=COLD_START_MAX_EVENTS),
        "warm": split.users_with_history(low=WARM_MIN_EVENTS),
    }
    for model in build_models_with(settings, params).values():
        results.extend(
            evaluate_segments(
                model, split, products, settings.eval_top_k, eval_segments,
                features=features, half_life_days=settings.recency_halflife_days,
                catalog=catalog, target="test",
            )
        )

    frame = results_to_frame(results)
    log.info("evaluation at k=%d:\n%s", settings.eval_top_k[1], frame[frame.k == 10].to_string())

    # ---- stage 5: persist --------------------------------------------------
    version = data_fingerprint(products, served_events)
    content = models["content"]
    bundle = ModelBundle(
        version=version,
        trained_at=utcnow(),
        models=models,
        features=features,
        interactions=interactions,
        products=products,
        item_matrix=getattr(content, "_item_matrix", None),
        params={
            "svd_factors": settings.svd_factors,
            "knn_top_k": settings.knn_top_k,
            "recency_halflife_days": settings.recency_halflife_days,
            "n_train_events": len(served_events),
            "n_val_events": len(split.val_events),
            "n_test_events": len(split.test_events),
            "n_eval_train_events": len(split.train_events),
            **params,
        },
        tuning=(tuning_result.to_dict() if tuning_result else {}),
    )
    save_bundle(bundle, metrics=[r.to_dict() for r in results], settings=settings)
    log.info("saved model bundle version=%s", version)
    return bundle, frame


def fusion_params(settings: Settings, tuning_result=None) -> dict:
    """Fusion hyperparameters, preferring tuned values when a search has run."""
    best = tuning_result.best_params if tuning_result else {}
    content_weight = float(best.get("content_weight", 0.4))
    collab_weight = float(best.get("collab_weight", 0.5))
    return {
        "content_weight": content_weight,
        "collab_weight": collab_weight,
        "popularity_weight": max(0.0, 1.0 - content_weight - collab_weight),
        "knn_weight": float(best.get("knn_weight", 0.5)),
    }


def build_models_with(settings: Settings, params: dict) -> dict:
    """A fresh, independent instance of every strategy."""
    from app.models.collaborative import PopularityRecommender

    content = ContentBasedRecommender(half_life_days=settings.recency_halflife_days)
    collaborative = CollaborativeRecommender(
        knn_top_k=settings.knn_top_k,
        n_factors=settings.svd_factors,
        knn_weight=params["knn_weight"],
    )
    return {
        "content": content,
        "collaborative": collaborative,
        "popularity": PopularityRecommender(),
        "hybrid": HybridRecommender(
            content=content,
            collaborative=collaborative,
            content_weight=params["content_weight"],
            collab_weight=params["collab_weight"],
            popularity_weight=params["popularity_weight"],
        ),
    }
