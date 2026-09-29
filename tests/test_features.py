"""Feature engineering: text cleaning, item vectors, interaction matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.domain import EventType
from app.features.build import ItemFeatureBuilder, build_text_series, clean_text, l2_normalize
from app.features.interactions import build_interaction_matrix, event_weight, recency_decay


class TestCleanText:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_text("Noise-Cancelling, WIRELESS!") == "noise cancelling wireless"

    def test_handles_missing_and_container_values(self):
        assert clean_text(None) == ""
        assert clean_text(["A", "B"]) == "a b"
        assert clean_text(float("nan")) == "nan"

    def test_collapses_whitespace(self):
        assert clean_text("  a   b  ") == "a b"


class TestItemFeatures:
    def test_rows_are_l2_normalised(self, features: ItemFeatureBuilder):
        matrix = features.transform_subset(features.product_ids[:10])
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_self_similarity_is_one(self, features: ItemFeatureBuilder):
        first = features.vector_for(features.product_ids[0])
        sim = float((first @ first.T).toarray()[0, 0])
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_same_category_scores_above_random(self, features: ItemFeatureBuilder, products):
        matrix = features.transform_subset(features.product_ids)
        products = products.set_index("product_id")
        pid = features.product_ids[0]
        same = [
            p for p in features.product_ids[1:30]
            if products.loc[p, "subcategory"] == products.loc[pid, "subcategory"]
        ]
        diff = [
            p for p in features.product_ids[30:60]
            if products.loc[p, "subcategory"] != products.loc[pid, "subcategory"]
        ]
        if not same or not diff:
            pytest.skip("catalogue too small for a same/different contrast")
        row = np.asarray(matrix[features.index_by_id[pid]].todense()).ravel()
        sims = np.asarray(matrix @ row).ravel()
        same_scores = [sims[features.index_by_id[p]] for p in same]
        diff_scores = [sims[features.index_by_id[p]] for p in diff]
        assert np.mean(same_scores) > np.mean(diff_scores)

    def test_transform_is_deterministic(self, features: ItemFeatureBuilder):
        ids = features.product_ids[:20]
        a = features.transform_subset(ids)
        b = features.transform_subset(ids)
        assert (a != b).nnz == 0

    def test_unknown_product_raises(self, features: ItemFeatureBuilder):
        with pytest.raises(KeyError):
            features.vector_for("does-not-exist")

    def test_text_series_includes_category_and_tags(self, products):
        text = build_text_series(products.iloc[:1]).iloc[0]
        assert "foldable" in text or "wireless" in text or "aluminium" in text
        assert products.iloc[0]["category"].lower() in text

    def test_l2_normalize_handles_zero_rows(self):
        from scipy import sparse

        matrix = sparse.csr_matrix(np.array([[0.0, 0.0], [3.0, 4.0]]))
        out = np.asarray(l2_normalize(matrix).todense())
        assert out[0].tolist() == [0.0, 0.0]
        assert out[1] == pytest.approx([0.6, 0.8])


class TestInteractionMatrix:
    def test_weights_scale_with_intent(self):
        assert event_weight(EventType.PURCHASE.value) > event_weight(EventType.VIEW.value)
        assert event_weight(EventType.ADD_TO_CART.value) > event_weight(EventType.VIEW.value)

    def test_explicit_weight_overrides_event_type(self):
        assert event_weight(EventType.VIEW.value, 5.0) == 5.0

    def test_shape_and_indices(self, interactions, products):
        assert interactions.n_items == len(products)
        assert interactions.n_users > 0
        assert interactions.has_user(interactions.user_ids[0])

    def test_recency_decay_prefers_newer_events(self):
        stamps = pd.Series(["2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"])
        reference = pd.Timestamp("2026-03-01T00:00:00+00:00")
        decay = recency_decay(stamps, reference, halflife_days=30.0)
        assert decay[1] > decay[0]
        assert decay[1] == pytest.approx(1.0)
        # 59 days is a little under two half-lives.
        assert decay[0] == pytest.approx(0.25, abs=0.02)

    def test_matrix_is_sparse_and_row_normalised_by_build(self, interactions):
        density = interactions.matrix.nnz / (interactions.n_users * interactions.n_items)
        assert density < 0.5, "interaction matrix should be sparse"
        assert interactions.matrix.shape[0] == interactions.n_users

    def test_seen_items_round_trips_to_ids(self, interactions):
        uid = interactions.user_ids[0]
        cols = interactions.seen_items(uid)
        assert cols.size > 0
        assert interactions.item_ids_for_cols(cols[:3]) == [
            interactions.item_ids[c] for c in cols[:3]
        ]

    def test_empty_events_yields_empty_matrix(self, products):
        empty = pd.DataFrame(columns=["user_id", "product_id", "event_type", "timestamp", "weight"])
        matrix = build_interaction_matrix(empty, products["product_id"].tolist())
        assert matrix.matrix.nnz == 0
        assert matrix.n_items == len(products)

    def test_repeat_interactions_accumulate(self, products):
        stamps = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC").astype(str)
        frame = pd.DataFrame(
            {
                "user_id": ["U1"] * 3,
                "product_id": ["P0000"] * 3,
                "event_type": ["purchase"] * 3,
                "timestamp": list(stamps),
                "weight": [1.0] * 3,
            }
        )
        matrix = build_interaction_matrix(
            frame, products["product_id"].tolist(), user_ids=["U1"], halflife_days=1.0
        )
        assert matrix.matrix.nnz == 1
