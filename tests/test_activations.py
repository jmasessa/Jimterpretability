"""Tests for activation extraction pipeline."""

import torch
import pytest

from jimterpretability.activations import extract_activations, _process_batch


def test_extract_activations_output_shape(mock_model, mock_sae, n_features):
    prompts = [f"prompt {i}" for i in range(7)]
    result = extract_activations(
        model=mock_model,
        sae=mock_sae,
        hook_point="blocks.0.hook_resid_post",
        prompts=prompts,
        batch_size=3,
        aggregate_seq="mean",
        device="cpu",
    )
    assert result.shape == (7, n_features), f"Expected (7, {n_features}), got {result.shape}"


@pytest.mark.parametrize("agg", ["mean", "max", "last"])
def test_aggregate_seq_modes(mock_model, mock_sae, n_features, agg):
    prompts = ["hello world", "foo bar baz"]
    result = extract_activations(
        model=mock_model,
        sae=mock_sae,
        hook_point="blocks.0.hook_resid_post",
        prompts=prompts,
        batch_size=2,
        aggregate_seq=agg,
    )
    assert result.shape == (2, n_features)


def test_invalid_aggregate_seq(mock_model, mock_sae):
    with pytest.raises(ValueError, match="aggregate_seq"):
        extract_activations(mock_model, mock_sae, "hook", ["hi"], aggregate_seq="invalid")


def test_single_prompt(mock_model, mock_sae, n_features):
    result = extract_activations(mock_model, mock_sae, "hook", ["one prompt"])
    assert result.shape == (1, n_features)


def test_padding_mask_mean_aggregation(d_model, n_features):
    """Mean aggregation must zero out padding positions."""
    import torch
    from unittest.mock import MagicMock

    # Build a model that returns known activations with a padding token at position 2
    model = MagicMock()
    model.tokenizer.pad_token_id = 0
    model.tokenizer.bos_token_id = 0

    # Tokens: real=1, real=2, pad=0
    tokens = torch.tensor([[1, 2, 0]])
    model.to_tokens.return_value = tokens

    # Hook activations: positions 0,1 have large values; position 2 (pad) has ones
    hook_acts = torch.zeros(1, 3, d_model)
    hook_acts[0, 0, :] = 2.0   # real
    hook_acts[0, 1, :] = 4.0   # real
    hook_acts[0, 2, :] = 999.0  # pad — should be masked

    model.run_with_cache.return_value = (None, {"hook_resid": hook_acts})

    sae = MagicMock()
    # encode: just return input reshaped to n_features (repeat to fill)
    def fake_encode(x):  # (N, d_model) -> (N, n_features)
        out = torch.zeros(x.shape[0], n_features)
        out[:, 0] = x[:, 0]  # carry first channel through
        return out
    sae.encode.side_effect = fake_encode

    result = _process_batch(model, sae, "hook_resid", ["a b"], "mean", "cpu")
    # Expected: mean of positions 0 and 1 only (padding excluded)
    # channel 0: (2.0 + 4.0) / 2 = 3.0
    assert abs(result[0, 0].item() - 3.0) < 0.01, f"Got {result[0, 0].item()}, expected 3.0"
    # Padding channel 0 value (999) must NOT appear
    assert result[0, 0].item() < 100
