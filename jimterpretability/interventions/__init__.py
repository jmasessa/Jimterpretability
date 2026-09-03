"""Intervention tools: activation steering, output interception, weight editing."""

from jimterpretability.interventions.steering import steer
from jimterpretability.interventions.interception import generate_with_interception
from jimterpretability.interventions.weight_edit import (
    apply_rank1_weight_edit,
    restore_weights,
    compute_feature_direction,
)

__all__ = [
    "steer",
    "generate_with_interception",
    "apply_rank1_weight_edit",
    "restore_weights",
    "compute_feature_direction",
]
