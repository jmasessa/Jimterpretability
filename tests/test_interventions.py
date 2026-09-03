"""Tests for interventions: steering, interception, weight editing."""

import torch
import pytest
from unittest.mock import MagicMock, patch

from jimterpretability.interventions.weight_edit import (
    apply_rank1_weight_edit,
    restore_weights,
    compute_feature_direction,
)


# ── Weight Edit ────────────────────────────────────────────────────────────────

def test_weight_edit_projects_out_direction(d_model):
    """After a full-scale edit, W_out @ feature_direction should be near zero."""
    W_out = torch.nn.Parameter(torch.randn(32, d_model))
    feature_direction = torch.randn(d_model)
    feature_direction = feature_direction / feature_direction.norm()

    model = MagicMock()
    model.blocks = [MagicMock()]
    model.blocks[0].mlp.W_out = W_out

    backup = apply_rank1_weight_edit(model, layer=0, feature_direction=feature_direction, scale=1.0)

    projections = model.blocks[0].mlp.W_out.data @ feature_direction
    assert projections.abs().max().item() < 1e-4, f"Max projection: {projections.abs().max().item()}"


def test_weight_edit_zero_scale_is_noop(d_model):
    """scale=0.0 must not change W_out at all."""
    W_out_original = torch.randn(32, d_model)
    W_out = torch.nn.Parameter(W_out_original.clone())
    model = MagicMock()
    model.blocks = [MagicMock()]
    model.blocks[0].mlp.W_out = W_out

    apply_rank1_weight_edit(model, 0, torch.randn(d_model), scale=0.0)
    assert torch.allclose(W_out.data, W_out_original)


def test_weight_edit_restore(d_model):
    """Restoring must recover the exact original weights."""
    W_out_original = torch.randn(32, d_model)
    W_out = torch.nn.Parameter(W_out_original.clone())
    model = MagicMock()
    model.blocks = [MagicMock()]
    model.blocks[0].mlp.W_out = W_out

    backup = apply_rank1_weight_edit(model, 0, torch.randn(d_model), scale=1.0)
    # Weights are changed
    assert not torch.allclose(W_out.data, W_out_original)

    restore_weights(model, 0, backup)
    assert torch.allclose(W_out.data, W_out_original), "Weights not restored correctly"


def test_compute_feature_direction_is_unit(n_features, d_model):
    sae = MagicMock()
    sae.W_dec = torch.randn(n_features, d_model)
    direction = compute_feature_direction(sae, feature_idx=5)
    assert abs(direction.norm().item() - 1.0) < 1e-5


# ── Interception ───────────────────────────────────────────────────────────────

def test_interception_triggers_when_above_threshold(d_model, n_features):
    """When the SAE returns activation above threshold, interception fires."""
    from jimterpretability.interventions.interception import generate_with_interception

    model = MagicMock()
    model.to_tokens.return_value = torch.zeros(1, 5, dtype=torch.long)
    model.tokenizer.eos_token_id = 1

    # run_with_hooks fires the hook with activations, SAE returns high activation for feature 3
    def run_with_hooks(tokens, fwd_hooks=None, return_type=None):
        B, T = tokens.shape
        for hook_point, fn in (fwd_hooks or []):
            acts = torch.randn(B, T, d_model)
            fn(acts, None)
        return torch.randn(B, T, 200)

    model.run_with_hooks.side_effect = run_with_hooks

    sae = MagicMock()
    # Feature 3 activation is always 5.0 — above any reasonable threshold
    def fake_encode(x):
        out = torch.zeros(x.shape[0], n_features)
        out[:, 3] = 5.0
        return out
    sae.encode.side_effect = fake_encode
    sae.W_enc = torch.randn(d_model, n_features)

    output, was_intercepted = generate_with_interception(
        model=model,
        sae=sae,
        hook_point="blocks.0.hook_resid_post",
        feature_idx=3,
        threshold=1.0,
        intercept_message="No Staffordshires!",
        prompt="Tell me about dogs",
        max_new_tokens=10,
    )
    assert was_intercepted
    assert output == "No Staffordshires!"


def test_interception_does_not_trigger_when_below_threshold(d_model, n_features):
    """When activation stays below threshold, generation completes normally."""
    from jimterpretability.interventions.interception import generate_with_interception

    model = MagicMock()
    tokens = torch.zeros(1, 5, dtype=torch.long)
    model.to_tokens.return_value = tokens
    model.tokenizer.eos_token_id = 1
    model.to_string.return_value = "normal output"

    call_count = {"n": 0}

    def run_with_hooks(toks, fwd_hooks=None, return_type=None):
        call_count["n"] += 1
        B, T = toks.shape
        for _, fn in (fwd_hooks or []):
            fn(torch.randn(B, T, d_model), None)
        # Return logits with argmax at token 5 (not EOS=1)
        logits = torch.zeros(B, T, 200)
        logits[:, :, 5] = 10.0
        return logits

    model.run_with_hooks.side_effect = run_with_hooks

    sae = MagicMock()
    def fake_encode(x):
        out = torch.zeros(x.shape[0], n_features)
        out[:, 3] = 0.1  # below threshold
        return out
    sae.encode.side_effect = fake_encode
    sae.W_enc = torch.randn(d_model, n_features)

    output, was_intercepted = generate_with_interception(
        model=model, sae=sae,
        hook_point="blocks.0.hook_resid_post",
        feature_idx=3, threshold=1.0,
        intercept_message="blocked",
        prompt="hello", max_new_tokens=3,
    )
    assert not was_intercepted
    assert call_count["n"] == 3  # ran all 3 steps


# ── Steering ───────────────────────────────────────────────────────────────────

def test_steer_calls_hook(mock_model, mock_sae):
    from jimterpretability.interventions.steering import steer
    from contextlib import contextmanager

    hooks_called = {"n": 0}

    @contextmanager
    def capture_hooks(fwd_hooks=None):
        for _, fn in (fwd_hooks or []):
            hooks_called["n"] += 1
        yield

    mock_model.hooks.side_effect = capture_hooks

    steer(
        model=mock_model,
        sae=mock_sae,
        feature_idx=0,
        alpha=5.0,
        hook_point="blocks.0.hook_resid_post",
        prompts=["hello"],
        max_new_tokens=5,
    )
    assert hooks_called["n"] == 1, "Expected the steering hook to be registered"
