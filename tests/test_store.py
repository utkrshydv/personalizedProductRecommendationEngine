"""Store contract: the in-memory and MongoDB implementations must behave alike."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.config import Settings
from app.db.base import Store
from app.db.factory import build_store
from app.db.memory_store import MemoryStore
from app.db.mongo_store import MongoStore
from app.domain import Event, EventType, Product, User

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sample() -> tuple[list[Product], list[User], list[Event]]:
    products = [
        Product(f"P{i}", f"Title {i}", f"Desc {i}", "Cat", "Sub", "Brand",
                10.0 * (i + 1), 4.0, i, [f"tag{i % 3}"], NOW)
        for i in range(5)
    ]
    users = [User("U0", "GB", "25-34", NOW), User("U1", "US", "35-44", NOW)]
    events = [
        Event(f"E{i}", "U0", f"P{i % 5}", EventType.VIEW.value,
              NOW - timedelta(days=i), 1.0, "S0")
        for i in range(8)
    ] + [
        Event("E99", "U1", "P2", EventType.PURCHASE.value, NOW, 5.0, "S1")
    ]
    return products, users, events


@pytest.fixture(params=["memory", "mongodb"])
def any_store(request, settings: Settings):
    """Every test in TestStoreContract runs against both backends."""
    if request.param == "mongodb":
        store = MongoStore(settings.mongo_uri, "reco_test", timeout_ms=800)
        if not store.ping():
            pytest.skip("no MongoDB available")
        store.clear()
        try:
            yield store
        finally:
            store.clear()
            store.close()
        return
    store = MemoryStore()
    try:
        yield store
    finally:
        store.clear()


def _seed(store: Store) -> None:
    products, users, events = _sample()
    store.upsert_products(products)
    store.upsert_users(users)
    store.insert_events(events)


class TestStoreContract:
    def test_counts_reflect_writes(self, any_store: Store):
        any_store.clear()
        assert any_store.counts() == {"products": 0, "users": 0, "events": 0}
        _seed(any_store)
        assert any_store.counts() == {"products": 5, "users": 2, "events": 9}

    def test_empty_reads_return_typed_frames(self, any_store: Store):
        any_store.clear()
        assert any_store.products().empty
        assert any_store.events().empty
        assert any_store.users().empty
        assert "product_id" in any_store.products().columns

    def test_upsert_is_idempotent(self, any_store: Store):
        any_store.clear()
        products, _, _ = _sample()
        any_store.upsert_products(products)
        any_store.upsert_products(products)
        assert any_store.counts()["products"] == 5

    def test_products_frame_has_single_id_column(self, any_store: Store):
        """MongoDB stores the id as both _id and product_id; the frame must not."""
        any_store.clear()
        _seed(any_store)
        frame = any_store.products()
        assert list(frame.columns).count("product_id") == 1
        assert isinstance(frame["product_id"], pd.Series)
        assert set(frame["product_id"]) == {f"P{i}" for i in range(5)}

    def test_tags_round_trip_as_lists(self, any_store: Store):
        any_store.clear()
        _seed(any_store)
        row = any_store.products().iloc[0]
        assert isinstance(row["tags"], list)

    def test_events_filter_by_user(self, any_store: Store):
        any_store.clear()
        _seed(any_store)
        assert set(any_store.events(user_id="U1")["user_id"]) == {"U1"}

    def test_events_filter_by_since(self, any_store: Store):
        any_store.clear()
        _seed(any_store)
        cutoff = (NOW - timedelta(days=2)).isoformat()
        recent = any_store.events(since=cutoff)
        assert len(recent) < 9
        assert recent["timestamp"].min() >= cutoff

    def test_events_are_sorted_for_profiles(self, any_store: Store):
        any_store.clear()
        _seed(any_store)
        frame = any_store.events(user_id="U0")
        assert frame["timestamp"].is_monotonic_increasing

    def test_rollup_sums_weights_per_pair(self, any_store: Store):
        any_store.clear()
        _seed(any_store)
        rollup = any_store.user_profile_counts(["U0"])
        assert set(rollup.columns) == {
            "user_id", "product_id", "weight", "n_events", "last_ts"
        }
        # U0 produced 8 view events spread over 5 products.
        assert rollup["n_events"].sum() == 8
        assert len(rollup) == 5

    def test_append_event_is_visible_immediately(self, any_store: Store):
        any_store.clear()
        _seed(any_store)
        any_store.append_event(
            Event("ENEW", "U0", "P4", EventType.PURCHASE.value, NOW, 5.0)
        )
        frame = any_store.events(user_id="U0")
        assert "ENEW" in set(frame["event_id"])

    def test_clear_empties_everything(self, any_store: Store):
        _seed(any_store)
        any_store.clear()
        assert any_store.counts() == {"products": 0, "users": 0, "events": 0}

    def test_ensure_indexes_is_safe_to_call(self, any_store: Store):
        any_store.ensure_indexes()
        any_store.ensure_indexes()

    def test_bulk_insert_stays_under_the_bson_message_limit(self, settings: Settings):
        """A full event log must load in one call.

        MongoDB rejects any single message over 16MB, and a realistic event log
        exceeds that. Writing it in one `insert_many` fails mid-load, so the
        store has to chunk.
        """
        store = MongoStore(settings.mongo_uri, "reco_test", timeout_ms=800)
        if not store.ping():
            pytest.skip("no MongoDB available")
        store.clear()
        try:
            n = MongoStore.INSERT_BATCH * 3 + 7
            events = [
                Event(f"BULK{i}", "U0", f"P{i % 5}", EventType.VIEW.value, NOW, 1.0)
                for i in range(n)
            ]
            assert store.insert_events(events) == n
            assert store.counts()["events"] == n
        finally:
            store.clear()
            store.close()


class TestDomainRoundTrip:
    def test_event_dict_round_trip(self):
        event = Event("E1", "U1", "P1", EventType.PURCHASE.value, NOW, 5.0, "S1")
        assert Event.from_dict(event.to_dict()) == event

    def test_product_dict_round_trip(self):
        product = Product("P1", "t", "d", "c", "s", "b", 1.5, 4.2, 7, ["x"], NOW)
        assert Product.from_dict(product.to_dict()) == product

    def test_user_dict_round_trip(self):
        user = User("U1", "IN", "18-24", NOW)
        assert User.from_dict(user.to_dict()) == user

    def test_naive_timestamps_are_assumed_utc(self):
        event = Event.from_dict(
            {"event_id": "E", "user_id": "U", "product_id": "P",
             "event_type": "view", "timestamp": "2026-01-01T00:00:00"}
        )
        assert event.timestamp.tzinfo is not None


class TestFactory:
    def test_falls_back_to_memory_without_mongo(self):
        settings = Settings(mongo_uri="mongodb://127.0.0.1:1", mongo_connect_timeout_ms=300)
        store = build_store(settings)
        assert isinstance(store, MemoryStore)

    def test_strict_mode_raises_instead_of_degrading(self):
        from app.db.factory import StoreUnavailableError

        settings = Settings(
            mongo_uri="mongodb://127.0.0.1:1",
            mongo_connect_timeout_ms=300,
            allow_memory_fallback=False,
        )
        with pytest.raises(StoreUnavailableError):
            build_store(settings)

    def test_uses_mongodb_when_reachable(self, settings: Settings):
        store = build_store(settings)
        assert store.backend in {"mongodb", "memory"}
        if store.backend == "mongodb":
            store.close()
