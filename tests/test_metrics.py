"""Metric correctness, checked against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from app.eval import metrics as M


def test_precision_recall_basic():
    recommended = ["a", "b", "c", "d"]
    relevant = {"a", "c", "e"}
    assert M.precision_at_k(recommended, relevant, 4) == pytest.approx(2 / 4)
    assert M.recall_at_k(recommended, relevant, 4) == pytest.approx(2 / 3)


def test_precision_uses_only_the_first_k():
    recommended = ["a", "x", "x", "c"]
    relevant = {"a", "c"}
    # At k=2 only "a" counts, so precision is 1/2 not 2/4.
    assert M.precision_at_k(recommended, relevant, 2) == pytest.approx(0.5)
    assert M.precision_at_k(recommended, relevant, 4) == pytest.approx(0.5)


def test_empty_recommendations_and_relevant_set():
    assert M.precision_at_k([], {"a"}, 10) == 0.0
    assert M.recall_at_k(["a"], set(), 10) == 0.0
    assert M.ndcg_at_k(["a"], set(), 10) == 0.0
    assert M.average_precision_at_k(["a"], set(), 10) == 0.0


def test_ndcg_matches_hand_computation():
    # Relevant items at ranks 2 and 3 out of 2 relevant available.
    recommended = ["x", "a", "b", "y"]
    relevant = {"a", "b"}
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert M.ndcg_at_k(recommended, relevant, 4) == pytest.approx(dcg / idcg)


def test_perfect_ranking_scores_one():
    relevant = {"a", "b", "c"}
    assert M.ndcg_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)
    assert M.average_precision_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)
    assert M.reciprocal_rank_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.0)


def test_worst_case_ranking_scores_low():
    relevant = {"x", "y", "z"}
    assert M.reciprocal_rank_at_k(["a", "b", "c"], relevant, 3) == 0.0
    assert M.hit_rate_at_k(["a", "b", "c"], relevant, 3) == 0.0


def test_average_precision_penalises_late_hits():
    early = M.average_precision_at_k(["a", "b", "x", "c"], {"a", "b", "c"}, 4)
    late = M.average_precision_at_k(["x", "y", "z", "a"], {"a", "b", "c"}, 4)
    assert early > late


def test_mrr_rewards_earlier_first_hit():
    relevant = {"b"}
    assert M.reciprocal_rank_at_k(["a", "b"], relevant, 2) == pytest.approx(0.5)
    assert M.reciprocal_rank_at_k(["b", "a"], relevant, 2) == pytest.approx(1.0)


def test_coverage_counts_distinct_items():
    lists = [["a", "b"], ["b", "c"]]
    assert M.coverage(lists, {"a", "b", "c", "d"}) == pytest.approx(3 / 4)
    assert M.coverage([], {"a"}) == 0.0


def test_novelty_is_negative_log_popularity():
    popularity = {"a": 0.5, "b": 0.25}
    value = M.novelty(["a", "b"], popularity)
    assert value == pytest.approx((-math.log2(0.5) - math.log2(0.25)) / 2)


def test_novelty_ignores_unknown_items():
    assert M.novelty(["missing"], {"a": 0.5}) == 0.0


def test_personalisation_extremes():
    assert M.personalisation([["a", "b"], ["c", "d"]]) == pytest.approx(1.0)
    assert M.personalisation([["a", "b"], ["a", "b"]]) == pytest.approx(0.0)
    assert M.personalisation([["a", "b"]]) == 0.0


def test_intra_list_diversity_is_zero_for_identical_items():
    # Item 0 is perfectly similar to itself, so a two-item list has diversity 0.
    sim = sparse.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert M.intra_list_diversity(["0", "1"], sim, {"0": 0, "1": 1}) == pytest.approx(0.0)


def test_intra_list_diversity_is_one_for_orthogonal_items():
    sim = sparse.csr_matrix(np.eye(2))
    assert M.intra_list_diversity(["0", "1"], sim, {"0": 0, "1": 1}) == pytest.approx(1.0)


def test_duplicate_hits_count_per_position():
    # These metrics score a *position*, not a distinct id. A recommender never
    # emits duplicates (rank_scores selects item indices, not ids), so the
    # distinction does not arise in practice - but the behaviour is pinned here
    # so a future change to ranking cannot silently inflate precision.
    assert M.precision_at_k(["a", "a", "a"], {"a"}, 3) == pytest.approx(1.0)
    assert M.recall_at_k(["a", "a", "a"], {"a"}, 3) == pytest.approx(1.0)
