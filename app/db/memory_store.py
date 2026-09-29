"""In-process store. Used when no MongoDB is reachable, and by tests."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

import pandas as pd

from app.db.base import EVENT_FIELDS, Store, _empty, _rollup
from app.domain import Event, Product, User


class MemoryStore(Store):
    """In-process store. Fast enough for dev, tests and demos."""

    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._products: dict[str, dict] = {}
        self._users: dict[str, dict] = {}
        self._events: list[dict] = []

    def close(self) -> None:
        return None

    def ensure_indexes(self) -> None:
        return None

    def upsert_products(self, products: Iterable[Product]) -> int:
        with self._lock:
            rows = [p.to_dict() for p in products]
            for row in rows:
                self._products[row["product_id"]] = row
            return len(rows)

    def upsert_users(self, users: Iterable[User]) -> int:
        with self._lock:
            rows = [u.to_dict() for u in users]
            for row in rows:
                self._users[row["user_id"]] = row
            return len(rows)

    def insert_events(self, events: Iterable[Event]) -> int:
        with self._lock:
            rows = [e.to_dict() for e in events]
            self._events.extend(rows)
            return len(rows)

    def append_event(self, event: Event) -> None:
        with self._lock:
            self._events.append(event.to_dict())

    def products(self) -> pd.DataFrame:
        with self._lock:
            if not self._products:
                return _empty("products")
            frame = pd.DataFrame(list(self._products.values()))
        if "tags" in frame:
            frame["tags"] = frame["tags"].apply(
                lambda v: list(v) if isinstance(v, list) else []
            )
        return frame.reset_index(drop=True)

    def users(self) -> pd.DataFrame:
        with self._lock:
            if not self._users:
                return _empty("users")
            return pd.DataFrame(list(self._users.values()))

    def events(self, since: str | None = None, user_id: str | None = None) -> pd.DataFrame:
        with self._lock:
            rows = list(self._events)
        frame = pd.DataFrame(rows, columns=list(EVENT_FIELDS)) if rows else _empty("events")
        if since is not None and not frame.empty:
            frame = frame[frame["timestamp"] >= since]
        if user_id is not None and not frame.empty:
            frame = frame[frame["user_id"] == user_id]
        if frame.empty:
            return frame.reset_index(drop=True)
        # MongoStore sorts by (user_id, timestamp); the two backends must agree
        # on ordering, because callers (profile rollups, recency weighting)
        # assume each user's events arrive oldest-first.
        order = pd.to_datetime(frame["timestamp"], utc=True, format="mixed").sort_values(
            kind="stable"
        ).index
        return frame.loc[order].reset_index(drop=True)

    def user_profile_counts(self, user_ids: Sequence[str] | None = None) -> pd.DataFrame:
        frame = self.events(user_id=user_ids[0] if user_ids else None)
        if user_ids and not frame.empty:
            frame = frame[frame["user_id"].isin(list(user_ids))]
        return _rollup(frame)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "products": len(self._products),
                "users": len(self._users),
                "events": len(self._events),
            }

    def clear(self) -> None:
        with self._lock:
            self._products.clear()
            self._users.clear()
            self._events.clear()
