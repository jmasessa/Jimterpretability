"""Inference-time activation steering via residual stream hook injection."""

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE


def steer(
    model: HookedTransformer,
    sae: SAE,
    feature_idx: int,
    alpha: float,
    hook_point: str,
    prompts: list[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> list[str]:
    """
    Run generation with a steering vector applied at every forward pass.

    alpha > 0 amplifies the feature; alpha < 0 suppresses it.
    The feature direction is the SAE decoder column: sae.W_dec[feature_idx].
    This direction is added to the residual stream at hook_point at every token position.

    Returns:
        List of generated strings (new tokens only, not the prompt).
    """
    feature_direction = sae.W_dec[feature_idx].detach().clone()  # (d_model,)

    def steering_hook(value: Tensor, hook) -> Tensor:
        # value: (batch, seq, d_model)
        return value + alpha * feature_direction.to(value.device, value.dtype)

    tokens = model.to_tokens(prompts, prepend_bos=True)  # (B, T_input)
    n_input = tokens.shape[1]

    with model.hooks(fwd_hooks=[(hook_point, steering_hook)]):
        with torch.no_grad():
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                verbose=False,
            )

    results = []
    for i in range(len(prompts)):
        new_tokens = output_tokens[i, n_input:]
        results.append(model.to_string(new_tokens))
    return results


def compute_baseline(
    model: HookedTransformer,
    prompts: list[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> list[str]:
    """Generate without any steering, for comparison."""
    tokens = model.to_tokens(prompts, prepend_bos=True)
    n_input = tokens.shape[1]

    with torch.no_grad():
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
        )

    results = []
    for i in range(len(prompts)):
        new_tokens = output_tokens[i, n_input:]
        results.append(model.to_string(new_tokens))
    return results
