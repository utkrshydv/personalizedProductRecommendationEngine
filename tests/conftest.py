"""Shared fixtures: a small, fast, deterministic dataset built once per session."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.db.memory_store import MemoryStore
from app.domain import Event, EventType, Product, User
from app.features.build import ItemFeatureBuilder
from app.features.interactions import build_interaction_matrix

SEED = 11


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        n_users=120,
        n_products=90,
        n_events=6000,
        random_seed=SEED,
        svd_factors=16,
        knn_top_k=30,
        content_top_k=30,
        eval_top_k=(5, 10),
        eval_holdout_per_user=3,
        tuning_val_size=1,
        tuning_max_users=60,
        allow_memory_fallback=True,
    )


@pytest.fixture(scope="session")
def products(settings: Settings) -> pd.DataFrame:
    """A small catalogue with real within-category variety in the text."""
    rng = np.random.default_rng(SEED)
    categories = ["Electronics", "Books", "Home & Kitchen", "Sports & Outdoors"]
    subcategories = ["Headphones", "Laptops", "Fiction", "Technical", "Cookware", "Fitness"]
    brands = ["Aurora", "Northwind", "Vertex", "Cobalt", "Summit"]
    vocab = [
        "wireless noise cancelling battery bluetooth usb",
        "portable speaker sound audio bass water resistant",
        "hardcover paperback illustrated bestseller author chapter",
        "guide tutorial reference practical examples complete",
        "non stick stainless steel dishwasher safe oven",
        "lightweight breathable adjustable grip training trail",
    ]
    modifiers = [
        "aluminium", "bamboo", "compact", "wireless", "rechargeable", "foldable",
        "insulated", "adjustable", "travel", "quiet", "smart", "refillable",
    ]
    rows = []
    for i in range(settings.n_products):
        category = categories[i % len(categories)]
        sub = subcategories[i % len(subcategories)]
        brand = brands[i % len(brands)]
        attrs = rng.choice(modifiers, size=3, replace=False).tolist()
        rows.append(
            {
                "product_id": f"P{i:04d}",
                "title": f"{brand} {attrs[0].title()} {sub}",
                "description": (
                    f"{sub} from {brand}. Features: {', '.join(attrs)}. "
                    f"{vocab[i % len(vocab)]}"
                ),
                "category": category,
                "subcategory": sub,
                "brand": brand,
                "price": round(float(np.exp(rng.normal(3.2, 0.6))), 2),
                "rating": round(float(rng.uniform(2.5, 5.0)), 2),
                "rating_count": int(rng.integers(0, 2000)),
                "tags": attrs[1:],
                "created_at": datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def events(products: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """A clickstream with genuine per-user taste, so the models have signal."""
    rng = np.random.default_rng(SEED + 1)
    n_items = len(products)
    n_cats = products["category"].nunique()
    cat_codes = products["category"].astype("category").cat.codes.to_numpy()
    log_price = np.log(products["price"].to_numpy())

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict] = []
    counter = 0
    for uid in range(settings.n_users):
        # Each user has a narrow taste: a preferred category and a price band.
        fav_cat = int(rng.integers(n_cats))
        price_pref = float(log_price.mean() + rng.normal(0, 0.4))
        # A stable "attribute cluster": 6% of items are this user's kind.
        favourite = rng.choice(n_items, size=max(3, n_items // 16), replace=False)
        fav_mask = np.zeros(n_items, dtype=bool)
        fav_mask[favourite] = True

        n_events = int(rng.poisson(settings.n_events / settings.n_users))
        for _ in range(max(4, n_events)):
            score = (
                3.0 * (cat_codes == fav_cat)
                + 4.0 * fav_mask
                - 1.5 * np.abs(log_price - price_pref)
                + rng.normal(0, 0.3, size=n_items)
            )
            probs = np.exp(score - score.max())
            probs /= probs.sum()
            pid = int(rng.choice(n_items, p=probs))
            roll = rng.random()
            etype = (
                EventType.PURCHASE if roll < 0.12
                else EventType.ADD_TO_CART if roll < 0.28
                else EventType.WISHLIST if roll < 0.45
                else EventType.VIEW
            )
            rows.append(
                {
                    "event_id": f"E{counter:07d}",
                    "user_id": f"U{uid:04d}",
                    "product_id": f"P{pid:04d}",
                    "event_type": etype.value,
                    "timestamp": (
                        now - timedelta(days=int(rng.integers(0, 120)),
                                       minutes=int(rng.integers(0, 1440)))
                    ).isoformat(),
                    "weight": 1.0,
                    "session_id": f"S{uid:04d}",
                }
            )
            counter += 1
    return pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def users(settings: Settings) -> pd.DataFrame:
    return pd.DataFrame(
        {"user_id": [f"U{i:04d}" for i in range(settings.n_users)],
         "country": "US", "age_band": "25-34"}
    )


@pytest.fixture(scope="session")
def features(products: pd.DataFrame) -> ItemFeatureBuilder:
    return ItemFeatureBuilder(min_df=1).fit(products)


@pytest.fixture(scope="session")
def interactions(events: pd.DataFrame, products: pd.DataFrame):
    return build_interaction_matrix(
        events, products["product_id"].tolist(), halflife_days=30.0
    )


@pytest.fixture()
def store(products: pd.DataFrame, events: pd.DataFrame, users: pd.DataFrame) -> MemoryStore:
    s = MemoryStore()
    s.upsert_products(Product.from_dict(r) for r in products.to_dict("records"))
    s.upsert_users(User.from_dict(r) for r in users.to_dict("records"))
    s.insert_events(Event.from_dict(r) for r in events.to_dict("records"))
    return s
