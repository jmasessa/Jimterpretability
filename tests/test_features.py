"""Tests for feature aggregation and ranking."""

import torch
import pytest

from jimterpretability.features import aggregate_features, rank_features, FeatureStats


def make_matrix(n_prompts=10, n_features=50):
    return torch.rand(n_prompts, n_features)


def test_returns_top_k():
    matrix = make_matrix()
    stats = aggregate_features(matrix, top_k=5)
    assert len(stats) == 5


def test_top_k_larger_than_features():
    matrix = make_matrix(n_features=10)
    stats = aggregate_features(matrix, top_k=50)
    assert len(stats) == 10


def test_frequency_computation():
    """Feature 5 activates in exactly 6/10 prompts — frequency should be 0.6."""
    matrix = torch.zeros(10, 20)
    matrix[:6, 5] = 1.0  # feature 5 active in first 6 prompts
    stats = aggregate_features(matrix, activation_threshold=0.0, top_k=20)
    # Find stats for feature 5
    stat5 = next((s for s in stats if s.feature_idx == 5), None)
    assert stat5 is not None
    assert abs(stat5.frequency - 0.6) < 1e-5, f"Expected 0.6, got {stat5.frequency}"


def test_max_magnitude():
    matrix = torch.zeros(10, 20)
    matrix[3, 7] = 42.0
    stats = aggregate_features(matrix, top_k=20)
    stat7 = next(s for s in stats if s.feature_idx == 7)
    assert abs(stat7.max_magnitude - 42.0) < 1e-4


def test_always_inactive_feature_has_zero_active_mean():
    matrix = torch.zeros(10, 5)
    stats = aggregate_features(matrix, top_k=5)
    for s in stats:
        assert s.active_mean == 0.0


def test_rank_by_mean_magnitude():
    stats = [
        FeatureStats(0, frequency=0.5, mean_magnitude=1.0, max_magnitude=2.0, active_mean=1.5),
        FeatureStats(1, frequency=0.5, mean_magnitude=3.0, max_magnitude=4.0, active_mean=3.5),
        FeatureStats(2, frequency=0.5, mean_magnitude=2.0, max_magnitude=3.0, active_mean=2.5),
    ]
    ranked = rank_features(stats, by="mean_magnitude")
    assert ranked[0].feature_idx == 1
    assert ranked[1].feature_idx == 2
    assert ranked[2].feature_idx == 0


def test_invalid_rank_by():
    with pytest.raises(ValueError):
        rank_features([], by="nonsense")
