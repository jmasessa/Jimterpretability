"""Rank-1 MLP weight editing to permanently suppress a feature direction."""

import uuid

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE


def compute_feature_direction(sae: SAE, feature_idx: int) -> Tensor:
    """
    Return the unit feature direction in residual stream space.

    The SAE decoder weight W_dec has shape (n_features, d_model).
    Column feature_idx is the direction this feature writes to the residual stream.
    """
    direction = sae.W_dec[feature_idx].clone().detach().float()
    return direction / direction.norm().clamp(min=1e-8)


def apply_rank1_weight_edit(
    model: HookedTransformer,
    layer: int,
    feature_direction: Tensor,
    scale: float = 1.0,
) -> dict:
    """
    Project the feature direction out of MLP W_out at the given layer.

    W_out has shape (d_mlp, d_model) in TransformerLens.
    For every row w_i, remove the component along feature_direction:
        W_out -= scale * outer(W_out @ d, d)

    Args:
        scale: 0.0 = no change, 1.0 = fully project out the direction.

    Returns:
        backup dict with {"edit_id", "layer", "W_out"} for later restoration.
    """
    W_out = model.blocks[layer].mlp.W_out  # Parameter, shape (d_mlp, d_model)
    d = feature_direction.to(W_out.device).to(W_out.dtype)
    d = d / d.norm().clamp(min=1e-8)

    backup = {
        "edit_id": str(uuid.uuid4()),
        "layer": layer,
        "W_out": W_out.data.clone(),
    }

    projections = W_out.data @ d              # (d_mlp,)
    rank1_update = projections.unsqueeze(-1) * d.unsqueeze(0)  # (d_mlp, d_model)
    W_out.data -= scale * rank1_update

    return backup


def restore_weights(
    model: HookedTransformer,
    layer: int,
    backup: dict,
) -> None:
    """Restore MLP W_out at layer from a backup dict created by apply_rank1_weight_edit."""
    model.blocks[layer].mlp.W_out.data.copy_(backup["W_out"])
