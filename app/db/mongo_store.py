"""MongoDB-backed store (pymongo).

Collections
    products  _id = product_id
    users     _id = user_id
    events    _id = event_id
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient, ReplaceOne
from pymongo.errors import PyMongoError

from app.db.base import EVENT_FIELDS, PRODUCT_FIELDS, USER_FIELDS, Store, _empty
from app.domain import Event, Product, User


def _normalise_id(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    """Guarantee exactly one `field` column, sourced from `_id` if needed.

    Documents store the id twice - as `_id` and as a readable field - so a plain
    rename yields two identically named columns and every later
    ``frame[field]`` returns a DataFrame instead of a Series.
    """
    if field in frame.columns:
        if "_id" in frame.columns:
            frame = frame.drop(columns=["_id"])
        return frame
    return frame.rename(columns={"_id": field})


class MongoStore(Store):
    backend = "mongodb"

    #: Documents per bulk write. Kept well under the 16MB max BSON message
    #: size so a large event log can never trip the server limit.
    UPSERT_BATCH = 1000
    INSERT_BATCH = 2000

    def __init__(self, uri: str, db_name: str, timeout_ms: int = 1500) -> None:
        self._client: MongoClient = MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            tz_aware=True,
        )
        self._db = self._client[db_name]
        self._products = self._db["products"]
        self._users = self._db["users"]
        self._events = self._db["events"]

    # -- lifecycle -------------------------------------------------------
    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    def close(self) -> None:
        self._client.close()

    def ensure_indexes(self) -> None:
        try:
            self._events.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
            self._events.create_index([("product_id", ASCENDING)])
            self._events.create_index([("timestamp", DESCENDING)])
            self._products.create_index([("category", ASCENDING)])
            self._products.create_index([("brand", ASCENDING)])
        except PyMongoError:
            # Index creation is an optimisation; a read-only or restricted
            # deployment should still be able to serve traffic.
            pass

    # -- writes ----------------------------------------------------------
    def upsert_products(self, products: Iterable[Product]) -> int:
        ops = [
            ReplaceOne({"_id": p.product_id}, {"_id": p.product_id, **p.to_dict()}, upsert=True)
            for p in products
        ]
        if not ops:
            return 0
        for i in range(0, len(ops), self.UPSERT_BATCH):
            self._products.bulk_write(ops[i : i + self.UPSERT_BATCH], ordered=False)
        return len(ops)

    def upsert_users(self, users: Iterable[User]) -> int:
        ops = [
            ReplaceOne({"_id": u.user_id}, {"_id": u.user_id, **u.to_dict()}, upsert=True)
            for u in users
        ]
        if not ops:
            return 0
        for i in range(0, len(ops), self.UPSERT_BATCH):
            self._users.bulk_write(ops[i : i + self.UPSERT_BATCH], ordered=False)
        return len(ops)

    def insert_events(self, events: Iterable[Event]) -> int:
        rows = []
        for e in events:
            row = e.to_dict()
            row["_id"] = row.pop("event_id")
            rows.append(row)
        if not rows:
            return 0
        # Chunked explicitly. A full event log easily exceeds MongoDB's 16MB
        # max BSON message size, and letting the driver discover that mid-load
        # aborts the whole insert.
        for i in range(0, len(rows), self.INSERT_BATCH):
            self._events.insert_many(rows[i : i + self.INSERT_BATCH], ordered=False)
        return len(rows)

    def append_event(self, event: Event) -> None:
        row = event.to_dict()
        row["_id"] = row.pop("event_id")
        self._events.replace_one({"_id": row["_id"]}, row, upsert=True)

    # -- reads -----------------------------------------------------------
    def products(self) -> pd.DataFrame:
        cursor = self._products.find({}, {f: 1 for f in PRODUCT_FIELDS}).sort("_id", ASCENDING)
        frame = pd.DataFrame(list(cursor))
        if frame.empty:
            return _empty("products")
        # Documents carry `_id` *and* a `product_id` field, both holding the same
        # value. Renaming would give the frame two identically-named columns, and
        # every downstream `frame["product_id"]` would then return a DataFrame.
        frame = _normalise_id(frame, "product_id")
        if "tags" in frame:
            frame["tags"] = frame["tags"].apply(
                lambda v: list(v) if isinstance(v, list) else []
            )
        return frame.reset_index(drop=True)

    def users(self) -> pd.DataFrame:
        cursor = self._users.find({}, {f: 1 for f in USER_FIELDS}).sort("_id", ASCENDING)
        frame = pd.DataFrame(list(cursor))
        if frame.empty:
            return _empty("users")
        return _normalise_id(frame, "user_id").reset_index(drop=True)

    def events(self, since: str | None = None, user_id: str | None = None) -> pd.DataFrame:
        query: dict = {}
        if since is not None:
            query["timestamp"] = {"$gte": since}
        if user_id is not None:
            query["user_id"] = user_id
        projection = {f: 1 for f in EVENT_FIELDS}
        cursor = self._events.find(query, projection).sort([("user_id", ASCENDING),
                                                            ("timestamp", ASCENDING)])
        frame = pd.DataFrame(list(cursor))
        if frame.empty:
            return _empty("events")
        return _normalise_id(frame, "event_id").reset_index(drop=True)

    def user_profile_counts(self, user_ids: Sequence[str] | None = None) -> pd.DataFrame:
        match: dict = {}
        if user_ids:
            match["user_id"] = {"$in": list(user_ids)}
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": {"user_id": "$user_id", "product_id": "$product_id"},
                    "weight": {"$sum": {"$ifNull": ["$weight", 1.0]}},
                    "n_events": {"$sum": 1},
                    "last_ts": {"$max": "$timestamp"},
                }
            },
            {"$sort": {"_id.user_id": 1, "last_ts": 1}},
        ]
        rows = list(self._events.aggregate(pipeline))
        if not rows:
            return pd.DataFrame(
                columns=["user_id", "product_id", "weight", "n_events", "last_ts"]
            )
        frame = pd.DataFrame(
            [
                {
                    "user_id": r["_id"]["user_id"],
                    "product_id": r["_id"]["product_id"],
                    "weight": float(r["weight"]),
                    "n_events": int(r["n_events"]),
                    "last_ts": r["last_ts"],
                }
                for r in rows
            ]
        )
        return frame.reset_index(drop=True)

    def counts(self) -> dict[str, int]:
        return {
            "products": self._products.estimated_document_count(),
            "users": self._users.estimated_document_count(),
            "events": self._events.estimated_document_count(),
        }

    def clear(self) -> None:
        self._products.delete_many({})
        self._users.delete_many({})
        self._events.delete_many({})
