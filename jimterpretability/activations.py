"""Activation extraction: run prompts through model + SAE to get sparse feature vectors."""

from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE

from jimterpretability.utils import batch_list


@dataclass
class ActivationBatch:
    prompts: list[str]
    hook_activations: Tensor   # (batch, seq, d_model)
    feature_acts: Tensor       # (batch, seq, n_features)
    attention_mask: Tensor     # (batch, seq) — 1 for real tokens, 0 for padding


def extract_activations(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompts: list[str],
    batch_size: int = 16,
    aggregate_seq: str = "mean",
    device: str = "cpu",
) -> Tensor:
    """
    Run all prompts through model + SAE, aggregate across sequence, return feature matrix.

    Returns:
        Tensor of shape (n_prompts, n_features) — one row per prompt.
    """
    if aggregate_seq not in ("mean", "max", "last"):
        raise ValueError(f"aggregate_seq must be 'mean', 'max', or 'last', got '{aggregate_seq}'")

    all_feature_acts: list[Tensor] = []

    for batch_prompts in batch_list(prompts, batch_size):
        batch_result = _process_batch(model, sae, hook_point, batch_prompts, aggregate_seq, device)
        all_feature_acts.append(batch_result)

    return torch.cat(all_feature_acts, dim=0)  # (n_prompts, n_features)


def _process_batch(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    prompts: list[str],
    aggregate_seq: str,
    device: str,
) -> Tensor:
    """Process one batch of prompts. Returns (batch_size, n_features)."""
    tokens = model.to_tokens(prompts, prepend_bos=True)  # (B, T)

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        # Some tokenizers (e.g. GPT-2) have no explicit pad token; BOS is used
        pad_id = model.tokenizer.bos_token_id
    attention_mask = (tokens != pad_id).float()  # (B, T)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_point,
            return_type=None,
        )
    hook_acts = cache[hook_point]  # (B, T, d_model)

    B, T, D = hook_acts.shape
    flat_acts = hook_acts.reshape(B * T, D)

    with torch.no_grad():
        flat_features = sae.encode(flat_acts)  # (B*T, n_features)

    feature_acts = flat_features.reshape(B, T, -1)  # (B, T, n_features)
    mask = attention_mask.unsqueeze(-1)              # (B, T, 1)

    if aggregate_seq == "mean":
        aggregated = (feature_acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif aggregate_seq == "max":
        masked = feature_acts.masked_fill(mask == 0, float("-inf"))
        aggregated = masked.max(dim=1).values
        # Replace -inf (fully masked rows) with 0
        aggregated = torch.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)
    else:  # "last"
        last_idx = attention_mask.long().sum(dim=1) - 1  # (B,)
        last_idx = last_idx.clamp(min=0)
        aggregated = feature_acts[torch.arange(B, device=feature_acts.device), last_idx, :]

    return aggregated  # (B, n_features)
