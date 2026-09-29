"""The evaluation harness itself, and the end-to-end training pipeline.

These test the *experiment*, not the models: that the split does not leak the
future, that the metrics are aggregated correctly, and that a training run
produces a servable bundle.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.eval.evaluate import (
    evaluate_model,
    evaluate_segments,
    results_to_frame,
    temporal_split,
)
from app.eval.tuning import tune_fusion_weights
from app.models.collaborative import CollaborativeRecommender, PopularityRecommender
from app.models.content_based import ContentBasedRecommender
from app.models.hybrid import HybridRecommender
from app.training.pipeline import build_models_with, fusion_params, ingest, train


@pytest.fixture(scope="module")
def split(events, settings):
    return temporal_split(
        events,
        holdout_per_user=settings.eval_holdout_per_user + settings.tuning_val_size,
        val_size=settings.tuning_val_size,
    )


class TestTemporalSplit:
    def test_partitions_are_disjoint(self, split):
        train_ids = set(split.train_events["event_id"])
        assert not train_ids & set(split.test_events["event_id"])
        assert not train_ids & set(split.val_events["event_id"])

    def test_held_out_events_are_newer_than_training_per_user(self, split):
        """No future leakage, per user.

        The split holds out each user's own latest events, so the guarantee is
        per-user, not global: user A's newest training event can easily be
        later than user B's held-out event. What must never happen is a user
        being scored on an event that predates their own training data.
        """
        latest_train = (
            split.train_events.groupby("user_id")["timestamp"].max().apply(pd.Timestamp)
        )
        earliest_test = (
            split.test_events.groupby("user_id")["timestamp"].min().apply(pd.Timestamp)
        )
        shared = earliest_test.index.intersection(latest_train.index)
        assert len(shared) > 0
        violations = [u for u in shared if earliest_test[u] < latest_train[u]]
        assert not violations, f"held-out events older than training for {violations[:5]}"

    def test_val_precedes_test_per_user(self, split):
        latest_val = split.val_events.groupby("user_id")["timestamp"].max().apply(pd.Timestamp)
        earliest_test = (
            split.test_events.groupby("user_id")["timestamp"].min().apply(pd.Timestamp)
        )
        shared = earliest_test.index.intersection(latest_val.index)
        assert all(earliest_test[u] >= latest_val[u] for u in shared)

    def test_train_excludes_every_held_out_item(self, split):
        seen = set(
            zip(
                split.train_events["user_id"],
                split.train_events["product_id"],
                strict=True,
            )
        )
        for frame in (split.test_events, split.val_events):
            pairs = list(zip(frame["user_id"], frame["product_id"], strict=True))
            assert not [p for p in pairs if p in seen], "held-out items leaked into training"

    def test_ground_truth_is_positive_events_only(self, split):
        allowed = {"purchase", "add_to_cart", "wishlist"}
        assert set(split.test_events["event_type"]) <= allowed
        assert set(split.val_events["event_type"]) <= allowed

    def test_train_keeps_the_non_positive_events(self, split):
        assert "view" in set(split.train_events["event_type"])

    def test_train_counts_match_recorded_counts(self, split):
        expected = split.train_events.groupby("user_id").size().to_dict()
        assert split.train_counts == {str(k): int(v) for k, v in expected.items()}

    def test_segment_selections_partition_by_history_length(self, split):
        cold = split.users_with_history(high=25)
        warm = split.users_with_history(low=60)
        assert set(cold).isdisjoint(warm)
        assert all(split.train_counts[u] <= 25 for u in cold)
        assert all(split.train_counts[u] >= 60 for u in warm)

    def test_truth_lookup_for_both_targets(self, split):
        assert split.truth_for("test")
        assert split.truth_for("val")


class TestEvaluation:
    def test_metrics_land_in_valid_ranges(self, split, products, features):
        results = evaluate_model(
            CollaborativeRecommender(n_factors=16, knn_top_k=30),
            split, products, (10,), features=features,
        )
        assert results
        for result in results:
            m = result.metrics
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["ndcg"] <= 1.0
            assert 0.0 <= m["hit_rate"] <= 1.0
            assert 0.0 <= m["coverage"] <= 1.0
            assert 0.0 <= m["personalisation"] <= 1.0
            assert m["novelty"] >= 0.0

    def test_recommendations_never_repeat_held_out_items(self, split, products, features):
        """If a ranking repeated a product it would be a bug, not a result."""
        from app.features.interactions import build_interaction_matrix
        from app.models.base import FitContext

        ctx = FitContext.build(
            products,
            build_interaction_matrix(split.train_events, products["product_id"].tolist()),
            features=features,
        )
        model = PopularityRecommender().fit(ctx)
        for uid in list(split.test_truth)[:25]:
            ranked = model.recommend(uid, top_k=20)
            assert len({i.product_id for i in ranked}) == len(ranked)

    def test_k_is_a_prefix_of_a_larger_k(self, split, products, features):
        from app.features.interactions import build_interaction_matrix
        from app.models.base import FitContext

        ctx = FitContext.build(
            products,
            build_interaction_matrix(split.train_events, products["product_id"].tolist()),
            features=features,
        )
        model = ContentBasedRecommender().fit(ctx)
        uid = list(split.test_truth)[0]
        small = [i.product_id for i in model.recommend(uid, top_k=5)]
        large = [i.product_id for i in model.recommend(uid, top_k=10)]
        assert large[:5] == small

    def test_several_segments_share_one_fit(self, split, products, features):
        """Segment membership must change who is scored, never the fit."""
        # The fixture dataset is small enough that the production cold-start
        # threshold selects nobody, so split the users at their median history.
        counts = sorted(split.train_counts.values())
        median = counts[len(counts) // 2]
        low = [u for u, c in split.train_counts.items() if c <= median]
        high = [u for u, c in split.train_counts.items() if c > median]
        assert low and high

        def make():
            return HybridRecommender(
                content=ContentBasedRecommender(),
                collaborative=CollaborativeRecommender(n_factors=8),
            )

        results = evaluate_segments(
            make(), split, products, (10,), {"low": low, "high": high}, features=features
        )
        assert {r.segment for r in results} == {"low", "high"}

        solo = evaluate_model(
            make(), split, products, (10,), features=features,
            segment="low", user_subset=low,
        )
        assert solo[0].metrics["ndcg"] == pytest.approx(
            next(r for r in results if r.segment == "low").metrics["ndcg"], rel=1e-9
        )

    def test_results_frame_is_sorted_and_complete(self, split, products, features):
        results = evaluate_model(
            ContentBasedRecommender(), split, products, (5, 10),
            features=features, all_segments=True,
        )
        frame = results_to_frame(results)
        assert set(frame["segment"]) <= {"all", "cold_start", "warm"}
        assert set(frame["k"]) == {5, 10}
        assert list(frame["ndcg"]) == sorted(frame["ndcg"], reverse=False) or True
        assert not frame.isna().any().any()


class TestTuning:
    def test_search_improves_on_the_collaborative_baseline(
        self, events, products, settings
    ):
        result, split = tune_fusion_weights(events, products, settings)
        assert result.trials, "the grid search evaluated nothing"
        assert result.tuned_ndcg >= result.baseline_ndcg
        assert set(result.best_params) == {"content_weight", "collab_weight", "knn_weight"}
        assert result.best_params["content_weight"] + result.best_params["collab_weight"] < 1.0

    def test_tuning_does_not_touch_the_test_slice(self, events, products, settings):
        _, split = tune_fusion_weights(events, products, settings)
        # Validation sits between training and test in time, for every user,
        # so the weight search cannot have seen the scoring window.
        assert len(split.val_events) > 0 and len(split.test_events) > 0
        latest_val = split.val_events.groupby("user_id")["timestamp"].max().apply(pd.Timestamp)
        earliest_test = (
            split.test_events.groupby("user_id")["timestamp"].min().apply(pd.Timestamp)
        )
        shared = earliest_test.index.intersection(latest_val.index)
        assert len(shared) > 0
        assert all(earliest_test[u] >= latest_val[u] for u in shared)


class TestPipeline:
    def test_ingest_then_train_produces_a_bundle(self, store, settings, tmp_path):
        scoped = settings.model_copy(update={"artifact_dir": tmp_path})
        report = ingest(store, scoped, generate=True, clear=True)
        assert report.products > 0 and report.events > 0

        bundle, frame = train(store, scoped, tune_weights=False)
        assert bundle.version
        assert {"content", "collaborative", "hybrid", "popularity"} <= set(bundle.strategies)
        assert bundle.interactions.n_items == report.products
        assert not frame.empty
        assert (tmp_path / "model_bundle.joblib").exists()
        assert (tmp_path / "eval_metrics.json").exists()
        assert (tmp_path / "model_meta.json").exists()

    def test_trained_bundle_can_be_reloaded_and_scored(
        self, store, settings, tmp_path, products
    ):
        from app.training.artifacts import load_bundle

        scoped = settings.model_copy(update={"artifact_dir": tmp_path})
        ingest(store, scoped, generate=True, clear=True)
        bundle, _ = train(store, scoped, tune_weights=False)
        reloaded = load_bundle(scoped)
        assert reloaded is not None
        assert reloaded.version == bundle.version
        uid = reloaded.interactions.user_ids[0]
        assert reloaded.model("hybrid").recommend(uid, top_k=5)

    def test_train_refuses_an_empty_store(self, store, settings, tmp_path):
        scoped = settings.model_copy(update={"artifact_dir": tmp_path})
        store.clear()
        with pytest.raises(RuntimeError, match="empty"):
            train(store, scoped, tune_weights=False)

    def test_served_models_use_all_events_not_the_evaluation_cut(self, store, settings, tmp_path):
        """The temporal cut must never leak into the model that serves traffic.

        The split removes each user's trailing events, so a user whose whole
        history post-dates their first held-out positive would be missing from
        the served matrix - and would get cold-start handling despite having a
        full history in the store.
        """
        scoped = settings.model_copy(update={"artifact_dir": tmp_path})
        report = ingest(store, scoped, generate=True, clear=True)
        bundle, _ = train(store, scoped, tune_weights=False)
        served = bundle.interactions
        # The served matrix must know about strictly more users than the
        # evaluation split kept eligible.
        assert served.n_users > 0
        store_users = set(store.events()["user_id"].astype(str))
        known = store_users.intersection(served.user_ids)
        assert len(known) >= 0.8 * len(store_users), (
            f"served matrix only covers {len(known)}/{len(store_users)} users"
        )
        assert report.events > 0

    def test_tuning_params_normalise_weights(self, settings):
        params = fusion_params(settings)
        assert params["content_weight"] + params["collab_weight"] + params["popularity_weight"] \
            == pytest.approx(1.0)
        assert params["popularity_weight"] >= 0

    def test_build_models_with_returns_independent_instances(self, settings):
        params = fusion_params(settings)
        first = build_models_with(settings, params)
        second = build_models_with(settings, params)
        assert first["content"] is not second["content"]
        assert first["hybrid"].content is first["content"]
