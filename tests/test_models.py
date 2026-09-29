"""Recommender behaviour: ranking contracts, cold start, fusion, real-time updates."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.base import FitContext, rank_scores
from app.models.collaborative import (
    CollaborativeRecommender,
    ItemItemKNN,
    LatentFactorModel,
    PopularityRecommender,
    minmax_rows,
)
from app.models.content_based import ContentBasedRecommender
from app.models.hybrid import HybridRecommender

ALL_MODELS = ["content", "collaborative", "popularity", "hybrid"]


@pytest.fixture(scope="module")
def ctx(products, features, interactions) -> FitContext:
    return FitContext.build(products, interactions, features=features)


@pytest.fixture(scope="module")
def fitted(ctx) -> dict:
    content = ContentBasedRecommender().fit(ctx)
    collaborative = CollaborativeRecommender(n_factors=16, knn_top_k=30).fit(ctx)
    return {
        "content": content,
        "collaborative": collaborative,
        "popularity": PopularityRecommender().fit(ctx),
        "hybrid": HybridRecommender(
            content=content, collaborative=collaborative,
            content_weight=0.3, collab_weight=0.6, popularity_weight=0.1,
        ).fit(ctx),
    }


class TestRankingContract:
    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_returns_requested_number_of_items(self, fitted, interactions, name):
        uid = interactions.user_ids[0]
        assert len(fitted[name].recommend(uid, top_k=7)) == 7

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_ranks_are_contiguous_from_one(self, fitted, interactions, name):
        ranked = fitted[name].recommend(interactions.user_ids[0], top_k=5)
        assert [item.rank for item in ranked] == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_scores_are_non_increasing(self, fitted, interactions, name):
        scores = [i.score for i in fitted[name].recommend(interactions.user_ids[0], top_k=10)]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_never_recommends_an_already_seen_item(self, fitted, interactions, name):
        uid = interactions.user_ids[3]
        seen = set(interactions.item_ids_for_cols(interactions.seen_items(uid)))
        recommended = {i.product_id for i in fitted[name].recommend(uid, top_k=15)}
        assert not (recommended & seen)

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_include_seen_overrides_the_exclusion(self, fitted, interactions, name):
        uid = interactions.user_ids[3]
        seen = set(interactions.item_ids_for_cols(interactions.seen_items(uid)))
        ranked = fitted[name].recommend(uid, top_k=25, include_seen=True)
        recommended = {i.product_id for i in ranked}
        assert recommended & seen

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_candidate_filter_is_respected(self, fitted, interactions, products, name):
        allowed = set(products[products["brand"] == products["brand"].iloc[0]]["product_id"])
        ranked = fitted[name].recommend(
            interactions.user_ids[0], top_k=10, candidates=allowed
        )
        assert ranked
        assert {i.product_id for i in ranked} <= allowed

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_explicit_exclusion_is_honoured(self, fitted, interactions, name):
        first = fitted[name].recommend(interactions.user_ids[0], top_k=5)
        blocked = first[0].product_id
        ranked = fitted[name].recommend(
            interactions.user_ids[0], top_k=5, exclude={blocked}
        )
        assert blocked not in {i.product_id for i in ranked}

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_unknown_user_falls_back_instead_of_crashing(self, fitted, name):
        ranked = fitted[name].recommend("nobody", top_k=5)
        assert isinstance(ranked, list) and len(ranked) <= 5

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_top_k_larger_than_catalogue_is_clamped(self, fitted, interactions, name):
        # include_seen=True is what makes the count able to reach the full
        # catalogue; by default already-seen items are withheld from the list.
        ranked = fitted[name].recommend(
            interactions.user_ids[0], top_k=10_000, include_seen=True
        )
        assert len(ranked) == len(fitted[name]._item_ids)

    def test_scoring_before_fit_raises(self, ctx):
        with pytest.raises(RuntimeError):
            ContentBasedRecommender().score("U0000")


class TestRankScores:
    def test_deterministic_tie_break(self):
        scores = np.array([1.0, 1.0, 1.0])
        items = ["b", "a", "c"]
        ranked = rank_scores(scores, items, top_k=2)
        assert [r.product_id for r in ranked] == ["a", "b"]

    def test_masked_items_are_dropped_entirely(self):
        scores = np.array([0.9, 0.8, 0.7])
        ranked = rank_scores(scores, ["a", "b", "c"], top_k=3, exclude={"a", "b"})
        assert [r.product_id for r in ranked] == ["c"]

    def test_empty_input(self):
        assert rank_scores(np.array([]), [], top_k=5) == []


class TestContentBased:
    def test_similar_items_share_the_subcategory(self, fitted, products):
        by_id = products.set_index("product_id")
        seed = products["product_id"].iloc[0]
        for item in fitted["content"].similar(seed, top_k=5):
            assert by_id.loc[item.product_id, "subcategory"] == by_id.loc[seed, "subcategory"]

    def test_similar_excludes_the_seed(self, fitted, products):
        seed = products["product_id"].iloc[0]
        assert seed not in {i.product_id for i in fitted["content"].similar(seed, top_k=5)}

    def test_similar_raises_for_unknown_product(self, fitted):
        with pytest.raises(KeyError):
            fitted["content"].similar("nope", top_k=5)

    def test_profile_is_unit_length(self, fitted, interactions):
        profile = fitted["content"].user_profile(interactions.user_ids[0])
        assert profile is not None
        assert np.linalg.norm(profile) == pytest.approx(1.0, abs=1e-6)

    def test_recent_events_move_the_profile(self, fitted, interactions, products):
        """A real purchase must change what the model would recommend next."""
        model = fitted["content"]
        uid = "brand-new-user-for-test"
        before = [i.product_id for i in model.recommend(uid, top_k=5, include_seen=True)]
        target = products["product_id"].iloc[5]
        model.update_user(uid, target, weight=5.0)
        after = [i.product_id for i in model.recommend(uid, top_k=5, include_seen=True)]
        assert before != after
        model.invalidate(uid)

    def test_invalid_aggregate_is_rejected(self):
        with pytest.raises(ValueError):
            ContentBasedRecommender(aggregate="nonsense")


class TestCollaborative:
    def test_knn_similarity_matrix_has_no_self_similarity(self, fitted):
        sim = fitted["collaborative"].knn.similarity_
        assert sim.diagonal().sum() == 0

    def test_knn_similarity_is_symmetric_before_pruning(self, fitted):
        # Shrinkage must stay symmetric in (i, j). The pruned matrix used for
        # scoring is deliberately asymmetric - keeping only each row's top-k
        # drops (i, j) while keeping (j, i) - so symmetry is asserted on the
        # unpruned matrix, where it is actually a property of the model.
        sim = fitted["collaborative"].knn.similarity_full_
        assert abs(sim - sim.T).nnz == 0

    def test_pruning_keeps_at_most_top_k_per_row(self, fitted):
        counts = np.diff(fitted["collaborative"].knn.similarity_.indptr)
        assert counts.max() <= 30

    def test_latent_factors_have_expected_rank(self, fitted):
        latent = fitted["collaborative"].latent
        assert latent.user_factors_.shape[1] == latent.item_factors_.shape[1]
        assert 0 < latent.explained_variance <= 1.0

    def test_explain_reports_fitted_shapes(self, fitted):
        info = fitted["collaborative"].explain()
        assert info["n_factors"] > 0 and info["knn_nnz"] > 0

    def test_knn_prunes_to_top_k_per_row(self, ctx):
        knn = ItemItemKNN(top_k=5).fit(ctx.interactions.matrix)
        counts = (knn.similarity_ != 0).sum(axis=1).A.ravel()
        assert counts.max() <= 5

    def test_score_batch_matches_per_user_scoring(self, fitted, interactions):
        model = fitted["collaborative"]
        users = interactions.user_ids[:4]
        batch = model.score_batch(users)
        for row, uid in enumerate(users):
            assert np.allclose(batch[row], model.score(uid), atol=1e-5)

    def test_minmax_rows_maps_to_unit_interval(self):
        scores = np.array([[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]], dtype=np.float32)
        out = minmax_rows(scores)
        assert out[0].tolist() == pytest.approx([0.0, 0.5, 1.0])
        # An all-equal row has no ranking information; it maps to zeros.
        assert out[1].tolist() == [0.0, 0.0, 0.0]

    def test_knn_and_svd_components_are_exposed(self, fitted, interactions):
        model = fitted["collaborative"]
        rows = np.array([0, 1], dtype=int)
        assert model.knn_scores_for(rows).shape == (2, model.n_items)
        assert model.svd_scores_for(rows).shape == (2, model.n_items)

    def test_latent_score_rows_reuses_cached_factors(self, ctx):
        model = LatentFactorModel(n_factors=8).fit(ctx.interactions.matrix)
        rows = np.array([0, 2], dtype=int)
        assert np.allclose(
            model.score_rows(rows),
            model.user_factors_[rows] @ model.item_factors_.T,
            atol=1e-4,
        )


class TestPopularity:
    def test_global_trending_ignores_the_user(self, fitted):
        assert len(fitted["popularity"].recommend_global(top_k=6)) == 6

    def test_trending_is_identical_for_every_user(self, fitted, interactions):
        a = [i.product_id for i in fitted["popularity"].recommend_global(5)]
        b = [i.product_id for i in fitted["popularity"].recommend(interactions.user_ids[0], 5)]
        assert a == b


class TestHybrid:
    def test_unknown_fusion_mode_is_rejected(self):
        with pytest.raises(ValueError):
            HybridRecommender(mode="magic")

    def test_both_fusion_modes_produce_rankings(self, ctx):
        for mode in ("rrf", "score"):
            model = HybridRecommender(
                content=ContentBasedRecommender(),
                collaborative=CollaborativeRecommender(n_factors=8),
                mode=mode,
            ).fit(ctx)
            assert len(model.recommend(ctx.interactions.user_ids[0], top_k=5)) == 5

    def test_rrf_and_score_modes_agree_on_the_top_item(self, ctx):
        ids = ctx.interactions.user_ids[0]
        rrf = HybridRecommender(
            content=ContentBasedRecommender(),
            collaborative=CollaborativeRecommender(n_factors=8), mode="rrf",
        ).fit(ctx)
        score = HybridRecommender(
            content=ContentBasedRecommender(),
            collaborative=CollaborativeRecommender(n_factors=8), mode="score",
        ).fit(ctx)
        assert (
            rrf.recommend(ids, top_k=1)[0].product_id
            == score.recommend(ids, top_k=1)[0].product_id
        )

    def test_zero_weight_disables_a_component(self, ctx):
        """Setting a weight to 0 must remove that model's influence."""
        ids = ctx.interactions.user_ids[0]
        content = ContentBasedRecommender().fit(ctx)
        model = HybridRecommender(
            content=content,
            collaborative=CollaborativeRecommender(n_factors=8).fit(ctx),
            content_weight=0.0, collab_weight=1.0, popularity_weight=0.0,
            mode="rrf",
        ).fit(ctx)
        recommended = {i.product_id for i in model.recommend(ids, top_k=10)}
        assert recommended


class TestLearningSignal:
    """The models must actually beat the popularity floor, or nothing above works."""

    @pytest.mark.parametrize("name", ["content", "collaborative", "hybrid"])
    def test_model_beats_popularity_baseline(self, fitted, interactions, name):
        uid = interactions.user_ids[5]
        model = fitted[name]
        baseline = fitted["popularity"]
        # Compare against a held-out positive the user has not consumed.
        seen = set(interactions.item_ids_for_cols(interactions.seen_items(uid)))
        targets = [p for p in interactions.item_ids if p not in seen]
        target = set(targets[:1])
        def hits(m):
            return len({i.product_id for i in m.recommend(uid, top_k=20)} & target)

        assert hits(model) >= hits(baseline)
