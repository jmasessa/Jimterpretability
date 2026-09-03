"""Shared fixtures: mock model, SAE, and session for unit tests (no model downloads)."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from jimterpretability.session import JimSession, SessionConfig


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def n_features():
    return 256


@pytest.fixture
def vocab_size():
    return 200


@pytest.fixture
def mock_model(d_model, vocab_size):
    """Fake HookedTransformer with the minimal API surface we use."""
    model = MagicMock()
    model.cfg.d_model = d_model
    model.cfg.n_layers = 2

    # to_tokens: returns (1, 10) token tensor; for list of prompts returns (B, 10)
    def fake_to_tokens(prompts, prepend_bos=True):
        if isinstance(prompts, str):
            return torch.zeros(1, 10, dtype=torch.long)
        return torch.zeros(len(prompts), 10, dtype=torch.long)

    model.to_tokens.side_effect = fake_to_tokens

    # tokenizer
    model.tokenizer.pad_token_id = 0
    model.tokenizer.bos_token_id = 0
    model.tokenizer.eos_token_id = 1

    # run_with_cache returns (None, {hook_point: activations})
    def fake_run_with_cache(tokens, names_filter=None, return_type=None):
        B, T = tokens.shape
        hook_acts = torch.randn(B, T, d_model)
        cache = {names_filter: hook_acts} if names_filter else {}
        return None, cache

    model.run_with_cache.side_effect = fake_run_with_cache

    # run_with_hooks returns logits
    def fake_run_with_hooks(tokens, fwd_hooks=None, return_type=None):
        B, T = tokens.shape
        # Fire all hooks
        for hook_point, fn in (fwd_hooks or []):
            fake_acts = torch.randn(B, T, d_model)
            fn(fake_acts, None)
        return torch.randn(B, T, vocab_size)

    model.run_with_hooks.side_effect = fake_run_with_hooks

    # generate returns input tokens + new tokens
    def fake_generate(tokens, max_new_tokens=10, temperature=1.0, verbose=False):
        B, T = tokens.shape
        new = torch.zeros(B, max_new_tokens, dtype=torch.long) + 5  # token id 5
        return torch.cat([tokens, new], dim=1)

    model.generate.side_effect = fake_generate

    # to_string
    model.to_string.return_value = "generated text"

    # hooks context manager
    from contextlib import contextmanager
    @contextmanager
    def fake_hooks(fwd_hooks=None):
        yield
    model.hooks.side_effect = fake_hooks

    # MLP W_out for weight edit tests
    for i in range(2):
        block = MagicMock()
        block.mlp.W_out = torch.nn.Parameter(torch.randn(128, d_model))
        model.blocks = [block, block]

    return model


@pytest.fixture
def mock_sae(d_model, n_features):
    """Fake SAE with sparse encode output."""
    sae = MagicMock()
    sae.W_dec = torch.randn(n_features, d_model)
    sae.W_enc = torch.randn(d_model, n_features)

    def fake_encode(x):  # x: (N, d_model) -> (N, n_features)
        out = torch.zeros(x.shape[0], n_features)
        out[:, :10] = torch.relu(torch.randn(x.shape[0], 10))
        return out

    sae.encode.side_effect = fake_encode
    return sae


@pytest.fixture
def session_config():
    return SessionConfig(
        model_name="gpt2-small",
        layer=0,
        hook_point="blocks.0.hook_resid_post",
        device="cpu",
        dtype="float32",
    )


@pytest.fixture
def mock_session(mock_model, mock_sae, session_config):
    return JimSession(
        session_id="test-session-id",
        config=session_config,
        model=mock_model,
        sae=mock_sae,
        neuronpedia_model_id="gpt2-sm",
        neuronpedia_layer_id="0-res-jb",
    )
