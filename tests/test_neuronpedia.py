"""Tests for the Neuronpedia client."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from jimterpretability.neuronpedia import NeuronpediaClient, NeuronpediaFeature, _parse_feature


SAMPLE_RESPONSE = {
    "explanations": [{"description": "dogs and canines"}],
    "activations": [
        {"tokens": ["the", " dog", " ran"], "values": [0.1, 2.5, 0.3]},
    ],
    "maxValue": 5.2,
}


@pytest.mark.asyncio
async def test_get_feature_parses_response():
    feature = _parse_feature(42, SAMPLE_RESPONSE)
    assert feature.feature_idx == 42
    assert feature.label == "dogs and canines"
    assert feature.max_activation == 5.2
    assert len(feature.activation_examples) == 1
    assert feature.activation_examples[0].tokens == ["the", " dog", " ran"]


@pytest.mark.asyncio
async def test_get_feature_http_call(respx_mock=None):
    """get_feature makes a GET to the correct URL."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = SAMPLE_RESPONSE

    client = NeuronpediaClient(model_id="gpt2-small", layer_id="8-res-jb")
    client._client.get = AsyncMock(return_value=mock_response)

    feature = await client.get_feature(42)
    client._client.get.assert_called_once_with("/feature/gpt2-small/8-res-jb/42")
    assert feature.label == "dogs and canines"
    await client.aclose()


@pytest.mark.asyncio
async def test_get_features_batch_returns_stubs_on_error():
    """Failed fetches return NeuronpediaFeature with None label (no crash)."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )
    mock_response.json.return_value = {}

    client = NeuronpediaClient(model_id="gpt2-small", layer_id="8-res-jb")
    client._client.get = AsyncMock(return_value=mock_response)

    results = await client.get_features_batch([1, 2, 3])
    assert len(results) == 3
    for r in results:
        assert r.label is None
    await client.aclose()


def test_parse_empty_response():
    feature = _parse_feature(99, {})
    assert feature.feature_idx == 99
    assert feature.label is None
    assert feature.activation_examples == []
    assert feature.max_activation is None
