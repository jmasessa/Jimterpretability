"""Feature aggregation and ranking across concept prompt batches."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FeatureStats:
    feature_idx: int
    frequency: float       # fraction of prompts where activation > threshold
    mean_magnitude: float  # mean activation across ALL prompts (including zeros)
    max_magnitude: float   # peak activation seen across all prompts
    active_mean: float     # mean activation only when the feature is active


@dataclass
class ConceptDiscoveryResult:
    concept_label: str
    n_prompts: int
    top_features: list[FeatureStats]
    raw_feature_matrix: Tensor | None = None  # (n_prompts, n_features), optional


def aggregate_features(
    feature_matrix: Tensor,
    activation_threshold: float = 0.0,
    top_k: int = 20,
) -> list[FeatureStats]:
    """
    Compute per-feature statistics across N prompts and return the top-k features.

    Args:
        feature_matrix: (n_prompts, n_features) tensor of aggregated feature activations
        activation_threshold: minimum activation to count as "active"
        top_k: how many features to return
    """
    n_prompts, n_features = feature_matrix.shape

    active_mask = feature_matrix > activation_threshold   # (N, F) bool

    frequency = active_mask.float().mean(dim=0)           # (F,)
    mean_magnitude = feature_matrix.mean(dim=0)           # (F,)
    max_magnitude = feature_matrix.max(dim=0).values      # (F,)

    active_sum = (feature_matrix * active_mask.float()).sum(dim=0)
    active_count = active_mask.float().sum(dim=0).clamp(min=1)
    active_mean = active_sum / active_count               # (F,) — 0 if never active

    # Rank by frequency primary, mean_magnitude secondary
    # torch.argsort descending: negate both
    sort_key = frequency * 1e6 + mean_magnitude           # simple composite score
    ranked_indices = torch.argsort(sort_key, descending=True)
    top_indices = ranked_indices[:top_k]

    stats = []
    for idx in top_indices:
        i = idx.item()
        stats.append(FeatureStats(
            feature_idx=i,
            frequency=frequency[i].item(),
            mean_magnitude=mean_magnitude[i].item(),
            max_magnitude=max_magnitude[i].item(),
            active_mean=active_mean[i].item() if active_mask[:, i].any() else 0.0,
        ))
    return stats


def rank_features(stats: list[FeatureStats], by: str = "frequency") -> list[FeatureStats]:
    """Re-sort an existing list of FeatureStats."""
    key_map = {
        "frequency": lambda s: s.frequency,
        "mean_magnitude": lambda s: s.mean_magnitude,
        "max_magnitude": lambda s: s.max_magnitude,
        "active_mean": lambda s: s.active_mean,
    }
    if by not in key_map:
        raise ValueError(f"'by' must be one of {list(key_map)}, got '{by}'")
    return sorted(stats, key=key_map[by], reverse=True)
