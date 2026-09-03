"""Neuronpedia REST API client for feature labels and activation examples."""

import asyncio
from dataclasses import dataclass, field

import httpx

NEURONPEDIA_BASE = "https://www.neuronpedia.org/api"


@dataclass
class ActivationExample:
    tokens: list[str]
    activations: list[float]


@dataclass
class NeuronpediaFeature:
    feature_idx: int
    label: str | None
    description: str | None
    activation_examples: list[ActivationExample] = field(default_factory=list)
    max_activation: float | None = None


class NeuronpediaClient:
    """
    Async client for the Neuronpedia feature API.

    model_id:  e.g. "gpt2-sm"
    layer_id:  e.g. "8-res-jb"
    """

    def __init__(
        self,
        model_id: str,
        layer_id: str,
        api_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.model_id = model_id
        self.layer_id = layer_id
        headers: dict[str, str] = {}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=NEURONPEDIA_BASE,
            headers=headers,
            timeout=timeout,
        )

    async def get_feature(self, feature_idx: int) -> NeuronpediaFeature:
        """Fetch label + activation examples for one feature."""
        url = f"/feature/{self.model_id}/{self.layer_id}/{feature_idx}"
        response = await self._client.get(url)
        response.raise_for_status()
        data = response.json()
        return _parse_feature(feature_idx, data)

    async def get_features_batch(
        self,
        feature_idxs: list[int],
        max_concurrent: int = 5,
    ) -> list[NeuronpediaFeature]:
        """Fetch multiple features with rate-limited concurrency."""
        sem = asyncio.Semaphore(max_concurrent)

        async def fetch_one(idx: int) -> NeuronpediaFeature:
            async with sem:
                try:
                    return await self.get_feature(idx)
                except httpx.HTTPStatusError:
                    # Return a stub for unlabeled / missing features
                    return NeuronpediaFeature(feature_idx=idx, label=None, description=None)

        return await asyncio.gather(*[fetch_one(i) for i in feature_idxs])

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()


def _parse_feature(feature_idx: int, data: dict) -> NeuronpediaFeature:
    """Parse a raw Neuronpedia API response into NeuronpediaFeature."""
    explanations = data.get("explanations") or []
    label = explanations[0].get("description") if explanations else None

    raw_activations = data.get("activations") or []
    examples = []
    for act in raw_activations:
        tokens = act.get("tokens", [])
        values = act.get("values", [])
        examples.append(ActivationExample(tokens=tokens, activations=values))

    return NeuronpediaFeature(
        feature_idx=feature_idx,
        label=label,
        description=label,
        activation_examples=examples,
        max_activation=data.get("maxValue"),
    )
