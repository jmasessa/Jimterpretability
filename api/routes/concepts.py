"""Concept discovery routes."""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from jimterpretability.activations import extract_activations
from jimterpretability.features import aggregate_features
from jimterpretability.neuronpedia import NeuronpediaClient
from api.schemas import ConceptDiscoverRequest, ConceptDiscoverResponse, FeatureStatResponse

router = APIRouter(tags=["concepts"])


def _get_session(session_id: str, request: Request):
    try:
        return request.app.state.registry.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.post("/sessions/{session_id}/concepts/discover", response_model=ConceptDiscoverResponse)
async def discover_concept(session_id: str, body: ConceptDiscoverRequest, request: Request):
    """
    Feed N prompts through the model + SAE and return top-k features by activation frequency.
    Optionally enriches results with Neuronpedia labels.
    """
    session = _get_session(session_id, request)

    # Run in a thread pool so the event loop isn't blocked by PyTorch
    loop = asyncio.get_running_loop()
    feature_matrix = await loop.run_in_executor(
        None,
        lambda: extract_activations(
            model=session.model,
            sae=session.sae,
            hook_point=session.config.hook_point,
            prompts=body.prompts,
            batch_size=body.batch_size,
            aggregate_seq=body.aggregate_seq,
            device=session.config.device,
        ),
    )

    stats = aggregate_features(feature_matrix, body.activation_threshold, body.top_k)

    # Fetch Neuronpedia labels if requested and if we have IDs
    np_labels: dict[int, str | None] = {}
    if body.include_neuronpedia and session.neuronpedia_model_id and session.neuronpedia_layer_id:
        async with NeuronpediaClient(
            model_id=session.neuronpedia_model_id,
            layer_id=session.neuronpedia_layer_id,
        ) as client:
            top_idxs = [s.feature_idx for s in stats]
            np_features = await client.get_features_batch(top_idxs)
            np_labels = {f.feature_idx: f.label for f in np_features}

    top_features = [
        FeatureStatResponse(
            feature_idx=s.feature_idx,
            frequency=s.frequency,
            mean_magnitude=s.mean_magnitude,
            max_magnitude=s.max_magnitude,
            active_mean=s.active_mean,
            neuronpedia_label=np_labels.get(s.feature_idx),
        )
        for s in stats
    ]

    return ConceptDiscoverResponse(
        concept_label=body.concept_label,
        n_prompts=len(body.prompts),
        session_id=session_id,
        top_features=top_features,
    )
