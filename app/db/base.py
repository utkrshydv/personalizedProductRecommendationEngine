"""Storage abstraction.

The service talks to a `Store`, not to a driver. Two implementations ship:
`MongoStore` (pymongo) and `MemoryStore` (dict-backed, used when no MongoDB is
reachable and `ALLOW_MEMORY_FALLBACK` is on). Both expose identical semantics
so swapping backends never changes the models or the API.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence

import pandas as pd

from app.domain import Event, Product, User

PRODUCT_FIELDS = (
    "product_id",
    "title",
    "description",
    "category",
    "subcategory",
    "brand",
    "price",
    "rating",
    "rating_count",
    "tags",
    "created_at",
)
USER_FIELDS = ("user_id", "country", "age_band", "created_at")
EVENT_FIELDS = (
    "event_id",
    "user_id",
    "product_id",
    "event_type",
    "timestamp",
    "weight",
    "session_id",
)


def _empty(kind: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list({"products": PRODUCT_FIELDS, "users": USER_FIELDS,
                                      "events": EVENT_FIELDS}[kind]))


class Store(abc.ABC):
    """Read/write access to users, products and interaction events."""

    backend: str = "unknown"

    # -- lifecycle -------------------------------------------------------
    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def ensure_indexes(self) -> None: ...

    # -- writes ----------------------------------------------------------
    @abc.abstractmethod
    def upsert_products(self, products: Iterable[Product]) -> int: ...

    @abc.abstractmethod
    def upsert_users(self, users: Iterable[User]) -> int: ...

    @abc.abstractmethod
    def insert_events(self, events: Iterable[Event]) -> int: ...

    @abc.abstractmethod
    def append_event(self, event: Event) -> None:
        """Insert a single event produced by live traffic."""

    # -- reads -----------------------------------------------------------
    @abc.abstractmethod
    def products(self) -> pd.DataFrame: ...

    @abc.abstractmethod
    def users(self) -> pd.DataFrame: ...

    @abc.abstractmethod
    def events(self, since: str | None = None, user_id: str | None = None) -> pd.DataFrame: ...

    @abc.abstractmethod
    def user_profile_counts(self, user_ids: Sequence[str] | None = None) -> pd.DataFrame:
        """Per-user rolled-up interaction counts, used to build the live profile."""

    @abc.abstractmethod
    def counts(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def clear(self) -> None: ...


def _rollup(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events into one weighted row per (user, product)."""
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "product_id", "weight", "n_events", "last_ts"])
    grp = (
        frame.assign(_w=frame["weight"].astype(float))
        .groupby(["user_id", "product_id"], as_index=False)
        .agg(weight=("_w", "sum"), n_events=("event_type", "size"), last_ts=("timestamp", "max"))
    )
    return grp.sort_values(["user_id", "last_ts"]).reset_index(drop=True)
