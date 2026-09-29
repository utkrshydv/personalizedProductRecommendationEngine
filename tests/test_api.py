"""API contract tests.

These run against the real app with an in-memory store and a freshly trained
bundle, so they exercise the same code path production traffic takes.
"""

from __future__ import annotations

import datetime

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.memory_store import MemoryStore
from app.deps import set_service
from app.domain import Event, Product, User
from app.main import create_app
from app.models.base import FitContext
from app.serving.service import RecommenderService
from app.training.artifacts import ModelBundle, save_bundle
from app.training.pipeline import build_models_with, fusion_params

PREFIX = "/api/v1"


@pytest.fixture(scope="module")
def api_store(products, events, users) -> MemoryStore:
    """Module-scoped so the trained bundle below can depend on it."""
    s = MemoryStore()
    s.upsert_products(Product.from_dict(r) for r in products.to_dict("records"))
    s.upsert_users(User.from_dict(r) for r in users.to_dict("records"))
    s.insert_events(Event.from_dict(r) for r in events.to_dict("records"))
    try:
        yield s
    finally:
        s.clear()


@pytest.fixture(scope="module")
def client(settings: Settings, products, api_store, features, interactions, tmp_path_factory):
    scoped = settings.model_copy(
        update={"artifact_dir": tmp_path_factory.mktemp("artifacts")}
    )

    ctx = FitContext.build(products, interactions, features=features)
    params = fusion_params(scoped)
    models = {name: model.fit(ctx) for name, model in build_models_with(scoped, params).items()}

    bundle = ModelBundle(
        version="testversion",
        trained_at=datetime.datetime.now(datetime.timezone.utc),
        models=models,
        features=features,
        interactions=interactions,
        products=products,
        item_matrix=models["content"]._item_matrix,
        params=params,
    )
    save_bundle(
        bundle, metrics=[{"model": "hybrid", "k": 10, "ndcg": 0.1}], settings=scoped
    )

    # Routes read settings through a cached accessor, so point that at the
    # temporary artifact directory for the duration of the module.
    import app.config as config_module

    original = config_module.get_settings
    config_module.get_settings = lambda: scoped
    try:
        set_service(RecommenderService(api_store, scoped))
        with TestClient(create_app(api_store)) as test_client:
            yield test_client, scoped
    finally:
        config_module.get_settings = original
        set_service(None)


class TestHealth:
    def test_reports_ready(self, client):
        test_client, _ = client
        response = test_client.get(f"{PREFIX}/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_version"] == "testversion"
        assert "hybrid" in body["models_loaded"]
        assert body["counts"]["products"] > 0

    def test_openapi_and_root(self, client):
        test_client, _ = client
        assert test_client.get("/").status_code == 200
        assert test_client.get("/openapi.json").status_code == 200


class TestRecommendations:
    def test_returns_ranked_products(self, client):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/recommendations", json={"user_id": "U0001", "top_k": 5}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 5
        assert [i["rank"] for i in body["items"]] == [1, 2, 3, 4, 5]
        assert body["strategy"] == "hybrid"
        assert body["latency_ms"] >= 0

    @pytest.mark.parametrize("strategy", ["hybrid", "content", "collaborative", "popularity"])
    def test_every_strategy_serves(self, client, strategy):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/recommendations",
            json={"user_id": "U0002", "top_k": 3, "strategy": strategy},
        )
        assert response.status_code == 200
        assert response.json()["items"]

    def test_unknown_strategy_is_rejected(self, client):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/recommendations", json={"user_id": "U0001", "strategy": "telepathy"}
        )
        # `Strategy` is a Literal, so the schema rejects it before the route runs.
        assert response.status_code == 422

    def test_price_filter_is_applied(self, client, products):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/recommendations",
            json={"user_id": "U0001", "top_k": 10, "max_price": 12.0},
        )
        assert response.status_code == 200
        assert all(i["price"] <= 12.0 for i in response.json()["items"])

    def test_category_filter_is_applied(self, client, products):
        test_client, _ = client
        category = products["category"].iloc[0]
        response = test_client.post(
            f"{PREFIX}/recommendations",
            json={"user_id": "U0001", "top_k": 5, "category": category},
        )
        assert response.status_code == 200
        assert all(i["category"] == category for i in response.json()["items"])

    def test_cold_start_user_is_flagged(self, client):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/recommendations", json={"user_id": "never-seen-user"}
        )
        assert response.status_code == 200
        assert response.json()["cold_start"] is True
        assert response.json()["items"], "cold start must still return something"

    def test_get_shortcut_matches_post(self, client):
        test_client, _ = client
        post = test_client.post(
            f"{PREFIX}/recommendations", json={"user_id": "U0003", "top_k": 4}
        ).json()
        get = test_client.get(
            f"{PREFIX}/users/U0003/recommendations?top_k=4&strategy=hybrid"
        ).json()
        assert [i["product_id"] for i in post["items"]] == [i["product_id"] for i in get["items"]]

    def test_top_k_is_bounded(self, client):
        test_client, _ = client
        assert test_client.get(
            f"{PREFIX}/users/U0001/recommendations?top_k=9999"
        ).status_code == 422

    def test_missing_user_id_is_a_validation_error(self, client):
        test_client, _ = client
        assert test_client.post(f"{PREFIX}/recommendations", json={}).status_code == 422


class TestSimilar:
    def test_similar_items(self, client, products):
        test_client, _ = client
        pid = products["product_id"].iloc[0]
        response = test_client.post(
            f"{PREFIX}/products/{pid}/similar", json={"top_k": 5}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["product_id"] == pid
        assert len(body["items"]) == 5
        assert pid not in {i["product_id"] for i in body["items"]}

    def test_unknown_product_is_404(self, client):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/products/P-does-not-exist/similar", json={"top_k": 3}
        )
        assert response.status_code == 404

    def test_content_strategy_similar_works(self, client, products):
        test_client, _ = client
        pid = products["product_id"].iloc[1]
        response = test_client.post(
            f"{PREFIX}/products/{pid}/similar", json={"top_k": 3, "strategy": "content"}
        )
        assert response.status_code == 200


class TestEvents:
    def test_event_is_accepted_and_persisted(self, client, products, api_store):
        test_client, _ = client
        pid = products["product_id"].iloc[3]
        response = test_client.post(
            f"{PREFIX}/events",
            json={"user_id": "U0004", "product_id": pid, "event_type": "purchase"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is True
        assert body["profile_updated"] is True
        stored = api_store.events(user_id="U0004")
        assert body["event_id"] in set(stored["event_id"])

    def test_view_does_not_trigger_a_profile_update(self, client, products):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/events",
            json={
                "user_id": "U0005", "product_id": products["product_id"].iloc[4],
                "event_type": "view",
            },
        )
        assert response.status_code == 202
        assert response.json()["profile_updated"] is False

    def test_event_changes_the_next_recommendation(self, client, products):
        test_client, _ = client
        uid = "brand-new-user"
        before = [
            i["product_id"]
            for i in test_client.post(
                f"{PREFIX}/recommendations", json={"user_id": uid, "top_k": 5}
            ).json()["items"]
        ]
        target = products["product_id"].iloc[7]
        test_client.post(
            f"{PREFIX}/events",
            json={"user_id": uid, "product_id": target, "event_type": "add_to_cart"},
        )
        after = [
            i["product_id"]
            for i in test_client.post(
                f"{PREFIX}/recommendations", json={"user_id": uid, "top_k": 5}
            ).json()["items"]
        ]
        assert before != after, "the recorded event must influence the next response"

    def test_invalid_event_type_is_rejected(self, client, products):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/events",
            json={
                "user_id": "U0006", "product_id": products["product_id"].iloc[0],
                "event_type": "teleported",
            },
        )
        assert response.status_code == 422

    def test_naive_timestamp_is_rejected(self, client, products):
        test_client, _ = client
        response = test_client.post(
            f"{PREFIX}/events",
            json={
                "user_id": "U0007", "product_id": products["product_id"].iloc[0],
                "timestamp": "2026-01-01T00:00:00",
            },
        )
        assert response.status_code == 422


class TestCatalogue:
    def test_pagination(self, client):
        test_client, _ = client
        page = test_client.get(f"{PREFIX}/products?limit=5&offset=0").json()
        nxt = test_client.get(f"{PREFIX}/products?limit=5&offset=5").json()
        assert len(page) == 5 and len(nxt) == 5
        assert {p["product_id"] for p in page}.isdisjoint({p["product_id"] for p in nxt})

    def test_facets(self, client):
        test_client, _ = client
        body = test_client.get(f"{PREFIX}/products/facets").json()
        assert body["categories"] and body["brands"] and body["tags"]
        assert len(body["price_range"]) == 2

    def test_search(self, client, products):
        test_client, _ = client
        title = products["title"].iloc[0]
        needle = title.split()[0]
        body = test_client.get(f"{PREFIX}/products?search={needle}").json()
        assert body
        assert all(needle.lower() in p["title"].lower() or
                   needle.lower() in p["description"].lower() for p in body)

    def test_get_product_and_404(self, client, products):
        test_client, _ = client
        pid = products["product_id"].iloc[0]
        assert test_client.get(f"{PREFIX}/products/{pid}").status_code == 200
        assert test_client.get(f"{PREFIX}/products/nope").status_code == 404

    def test_trending(self, client):
        test_client, _ = client
        body = test_client.get(f"{PREFIX}/trending?top_k=4").json()
        assert len(body) == 4


class TestProfileAndAdmin:
    def test_profile_for_known_user(self, client):
        test_client, _ = client
        body = test_client.get(f"{PREFIX}/users/U0001/profile").json()
        assert body["exists"] is True
        assert body["n_events"] > 0
        assert body["event_type_counts"]

    def test_profile_for_unknown_user(self, client):
        test_client, _ = client
        body = test_client.get(f"{PREFIX}/users/ghost/profile").json()
        assert body["exists"] is False
        assert body["cold_start"] is True

    def test_model_info(self, client):
        test_client, _ = client
        body = test_client.get(f"{PREFIX}/models").json()
        assert {m["strategy"] for m in body} >= {"hybrid", "content", "collaborative"}
        assert all(m["version"] == "testversion" for m in body)

    def test_metrics_endpoint(self, client):
        test_client, _ = client
        response = test_client.get(f"{PREFIX}/metrics")
        assert response.status_code == 200
        assert response.json()["rows"]

    def test_reload(self, client):
        test_client, _ = client
        assert test_client.post(f"{PREFIX}/admin/reload").json()["reloaded"] is True
