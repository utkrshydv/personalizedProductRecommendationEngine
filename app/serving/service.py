"""Serving layer.

Owns the loaded model bundle and turns API requests into ranked products. This
is the only place that knows about both the models and the store, so the routes
stay thin.

Two behaviours matter for production:

* **Real-time updates.** A new event is written to the store *and* folded
  straight into the in-memory content profile, so the very next request already
  reflects it. No retrain, no reload.
* **Bounded staleness.** The bundle is cached with a TTL so a retrain performed
  by another process is picked up without a restart.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.db.base import Store
from app.domain import EVENT_WEIGHTS, Event, EventType
from app.models.base import ScoredItem
from app.schemas import (
    ProductOut,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
)
from app.training.artifacts import ModelBundle, load_bundle, utcnow

log = logging.getLogger(__name__)


@dataclass
class CatalogFilters:
    category: str | None = None
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    tags: frozenset[str] = frozenset()

    def is_empty(self) -> bool:
        return not any(
            [self.category, self.brand, self.min_price is not None,
             self.max_price is not None, self.tags]
        )

    def apply(self, products: pd.DataFrame) -> set[str]:
        frame = products
        if self.category:
            frame = frame[frame["category"] == self.category]
        if self.brand:
            frame = frame[frame["brand"] == self.brand]
        price = pd.to_numeric(frame["price"], errors="coerce")
        if self.min_price is not None:
            frame = frame[price >= self.min_price]
        if self.max_price is not None:
            frame = frame[price <= self.max_price]
        if self.tags:
            wanted = {t.lower() for t in self.tags}
            frame = frame[
                frame["tags"].apply(
                    lambda tags: bool(wanted & {str(t).lower() for t in tags or []})
                )
            ]
        return set(frame["product_id"].astype(str))


class RecommenderService:
    """Loads the bundle, serves recommendations, absorbs live events."""

    def __init__(self, store: Store, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._bundle: ModelBundle | None = None
        self._loaded_at: float = 0.0
        self._load_error: str | None = None

    # -- bundle lifecycle --------------------------------------------------
    def bundle(self, force: bool = False) -> ModelBundle | None:
        with self._lock:
            stale = (
                self._bundle is None
                or force
                or (time.monotonic() - self._loaded_at) > self.settings.model_cache_ttl_seconds
            )
            if stale:
                self._bundle = load_bundle(self.settings)
                self._loaded_at = time.monotonic()
                if self._bundle is None:
                    self._load_error = "no trained model bundle found; run the training pipeline"
                else:
                    self._load_error = None
            return self._bundle

    def reload(self) -> bool:
        self.bundle(force=True)
        return self._bundle is not None

    @property
    def model_version(self) -> str:
        bundle = self.bundle()
        return bundle.version if bundle else "unfitted"

    def status(self) -> dict:
        bundle = self.bundle()
        try:
            counts = self.store.counts()
        except Exception:
            counts = {}
        return {
            "store_backend": self.store.backend,
            "model_version": bundle.version if bundle else None,
            "models_loaded": bundle.strategies if bundle else [],
            "counts": counts,
            "load_error": self._load_error,
        }

    # -- catalogue ---------------------------------------------------------
    def products(
        self,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        brand: str | None = None,
        search: str | None = None,
    ) -> list[ProductOut]:
        bundle = self._require()
        frame = bundle.products
        if category:
            frame = frame[frame["category"] == category]
        if brand:
            frame = frame[frame["brand"] == brand]
        if search:
            needle = search.lower()
            frame = frame[
                frame["title"].str.lower().str.contains(needle, na=False)
                | frame["description"].str.lower().str.contains(needle, na=False)
            ]
        page = frame.iloc[offset : offset + limit]
        return [ProductOut(**row) for row in _records(page)]

    def product(self, product_id: str) -> ProductOut | None:
        bundle = self._require()
        match = bundle.products[bundle.products["product_id"] == product_id]
        if match.empty:
            return None
        return ProductOut(**_records(match)[0])

    def facets(self) -> dict:
        bundle = self._require()
        frame = bundle.products
        return {
            "categories": sorted(frame["category"].dropna().unique().tolist()),
            "subcategories": sorted(frame["subcategory"].dropna().unique().tolist()),
            "brands": sorted(frame["brand"].dropna().unique().tolist()),
            "tags": sorted({str(t) for tags in frame["tags"] for t in (tags or [])}),
            "price_range": [
                float(pd.to_numeric(frame["price"], errors="coerce").min()),
                float(pd.to_numeric(frame["price"], errors="coerce").max()),
            ],
        }

    def trending(self, top_k: int = 10) -> list[RecommendationItem]:
        bundle = self._require()
        model = bundle.model("popularity")
        items = model.recommend_global(top_k=top_k)
        return self._hydrate(items)

    # -- recommendations ---------------------------------------------------
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        t0 = time.perf_counter()
        bundle = self._require()
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        model = bundle.model(request.strategy)

        interactions = bundle.interactions
        has_history = (
            bool(interactions.has_user(request.user_id))
            and interactions.seen_items(request.user_id).size > 0
        ) if interactions is not None else False

        exclude = set(request.exclude_product_ids)
        candidates: set[str] | None = None
        filters = CatalogFilters(
            category=request.category,
            brand=request.brand,
            min_price=request.min_price,
            max_price=request.max_price,
            tags=frozenset(t.lower() for t in request.tags),
        )
        if not filters.is_empty():
            candidates = filters.apply(bundle.products)

        ranked = model.recommend(
            request.user_id,
            top_k=top_k,
            exclude=exclude or None,
            candidates=candidates,
            include_seen=request.include_seen,
        )
        items = self._hydrate(ranked)
        return RecommendationResponse(
            user_id=request.user_id,
            strategy=request.strategy,
            top_k=top_k,
            items=items,
            count=len(items),
            model_version=bundle.version,
            generated_at=utcnow(),
            cold_start=not has_history,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

    def similar(
        self, product_id: str, top_k: int, strategy: str = "hybrid"
    ) -> list[RecommendationItem]:
        bundle = self._require()
        model = bundle.model(strategy)
        if not hasattr(model, "similar"):
            raise ValueError(f"strategy {strategy!r} does not support similar-item lookups")
        return self._hydrate(model.similar(product_id, top_k=top_k))

    # -- profiles ----------------------------------------------------------
    def profile(self, user_id: str) -> UserProfile:
        bundle = self._require(allow_unfitted=True)
        interactions = bundle.interactions if bundle else None
        counts: dict[str, int] = {}
        if interactions is not None and interactions.has_user(user_id):
            seen = interactions.seen_items(user_id)
            counts = {"interacted_items": len(seen)}

        try:
            events = self.store.events(user_id=user_id)
        except Exception as exc:
            log.warning("could not read events for %s: %s", user_id, exc)
            events = pd.DataFrame()

        if events.empty:
            return UserProfile(user_id=user_id, exists=False, cold_start=True)

        by_type = events["event_type"].value_counts().to_dict()
        merged = bundle.products.merge(events, on="product_id", how="inner")
        top_categories = (
            merged["category"].value_counts().head(5).items() if not merged.empty else []
        )
        top_brands = merged["brand"].value_counts().head(5).items() if not merged.empty else []
        price = (
            pd.to_numeric(merged["price"], errors="coerce").dropna()
            if not merged.empty
            else pd.Series(dtype=float)
        )

        return UserProfile(
            user_id=user_id,
            exists=True,
            n_events=len(events),
            n_distinct_products=int(events["product_id"].nunique()),
            event_type_counts={str(k): int(v) for k, v in by_type.items()},
            top_categories=[(str(k), int(v)) for k, v in top_categories],
            top_brands=[(str(k), int(v)) for k, v in top_brands],
            price_range=(float(price.min()), float(price.max())) if len(price) else None,
            cold_start=not counts,
        )

    # -- live events -------------------------------------------------------
    def record_event(
        self,
        user_id: str,
        product_id: str,
        event_type: str,
        session_id: str | None = None,
        timestamp: datetime | None = None,
        weight: float | None = None,
    ) -> Event:
        """Persist an event and fold it into the live content profile."""
        bundle = self.bundle()
        resolved_weight = weight or EVENT_WEIGHTS.get(event_type, 1.0)
        event = Event(
            event_id=uuid.uuid4().hex,
            user_id=user_id,
            product_id=product_id,
            event_type=event_type,
            timestamp=timestamp or datetime.now(timezone.utc),
            weight=float(resolved_weight),
            session_id=session_id,
        )
        self.store.append_event(event)

        # Incremental profile update: cheap, and it makes the next request
        # reflect this event without a retrain.
        if bundle is not None and event_type != EventType.VIEW.value:
            content = bundle.models.get("content")
            if content is not None and event_type in {EventType.WISHLIST.value,
                                                      EventType.ADD_TO_CART.value,
                                                      EventType.PURCHASE.value,
                                                      EventType.RATING.value}:
                try:
                    content.update_user(user_id, product_id, event.weight)
                except Exception as exc:
                    log.warning("incremental profile update failed for %s: %s", user_id, exc)
                    content.invalidate(user_id)
        return event

    # -- helpers -----------------------------------------------------------
    def _require(self, allow_unfitted: bool = False) -> ModelBundle:
        bundle = self.bundle()
        if bundle is None and not allow_unfitted:
            raise ModelNotReadyError(
                "no trained model available. Run `python -m scripts.train` (or POST "
                f"{self.settings.api_prefix}/admin/train) to build one."
            )
        return bundle  # type: ignore[return-value]

    def _hydrate(self, ranked: list[ScoredItem]) -> list[RecommendationItem]:
        bundle = self._require(allow_unfitted=True)
        if not ranked or bundle is None:
            return []
        products = bundle.products.set_index("product_id")
        interactions = bundle.interactions
        totals = (
            np.asarray(interactions.matrix.sum(axis=0)).ravel()
            if interactions is not None
            else np.zeros(len(bundle.products))
        )
        total = totals.sum() or 1.0
        item_index = {p: i for i, p in enumerate(interactions.item_ids)} if interactions else {}

        out: list[RecommendationItem] = []
        for item in ranked:
            if item.product_id not in products.index:
                continue
            row = products.loc[item.product_id]
            idx = item_index.get(item.product_id)
            out.append(
                RecommendationItem(
                    product_id=item.product_id,
                    title=str(row["title"]),
                    description=str(row["description"]),
                    category=str(row["category"]),
                    subcategory=str(row["subcategory"]),
                    brand=str(row["brand"]),
                    price=float(row["price"]),
                    rating=float(row["rating"]),
                    rating_count=int(row["rating_count"]),
                    tags=[str(t) for t in (row["tags"] or [])],
                    popularity=(float(totals[idx] / total) if idx is not None else 0.0),
                    score=round(item.score, 6),
                    rank=item.rank,
                    reason=item.reason,
                )
            )
        return out


class ModelNotReadyError(RuntimeError):
    pass


def _records(frame: pd.DataFrame) -> list[dict]:
    """Dataframe rows -> JSON-safe dicts (numpy scalars and NaN included)."""
    if frame.empty:
        return []
    out = []
    for row in frame.to_dict(orient="records"):
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating,)):
                cleaned[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, list):
                cleaned[key] = [str(v) for v in value]
            else:
                cleaned[key] = value
        out.append(cleaned)
    return out
