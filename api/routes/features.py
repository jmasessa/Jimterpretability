"""Feature inspection routes: look up Neuronpedia labels and examples."""

from fastapi import APIRouter, HTTPException, Request

from jimterpretability.neuronpedia import NeuronpediaClient
from api.schemas import FeatureDetailResponse, FeatureBatchRequest, ActivationExampleResponse

router = APIRouter(tags=["features"])


def _get_session(session_id: str, request: Request):
    try:
        return request.app.state.registry.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.get("/sessions/{session_id}/features/{feature_idx}", response_model=FeatureDetailResponse)
async def get_feature(session_id: str, feature_idx: int, request: Request):
    session = _get_session(session_id, request)

    if not session.neuronpedia_model_id or not session.neuronpedia_layer_id:
        raise HTTPException(
            status_code=422,
            detail=f"Neuronpedia not available for model '{session.config.model_name}'",
        )

    async with NeuronpediaClient(
        model_id=session.neuronpedia_model_id,
        layer_id=session.neuronpedia_layer_id,
    ) as client:
        try:
            feature = await client.get_feature(feature_idx)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Neuronpedia error: {exc}") from exc

    return FeatureDetailResponse(
        feature_idx=feature.feature_idx,
        neuronpedia_label=feature.label,
        description=feature.description,
        activation_examples=[
            ActivationExampleResponse(tokens=ex.tokens, activations=ex.activations)
            for ex in feature.activation_examples
        ],
        max_activation=feature.max_activation,
    )


@router.post("/sessions/{session_id}/features/inspect-batch", response_model=list[FeatureDetailResponse])
async def inspect_batch(session_id: str, body: FeatureBatchRequest, request: Request):
    session = _get_session(session_id, request)

    if not session.neuronpedia_model_id or not session.neuronpedia_layer_id:
        raise HTTPException(
            status_code=422,
            detail=f"Neuronpedia not available for model '{session.config.model_name}'",
        )

    async with NeuronpediaClient(
        model_id=session.neuronpedia_model_id,
        layer_id=session.neuronpedia_layer_id,
    ) as client:
        features = await client.get_features_batch(body.feature_idxs)

    return [
        FeatureDetailResponse(
            feature_idx=f.feature_idx,
            neuronpedia_label=f.label,
            description=f.description,
            activation_examples=[
                ActivationExampleResponse(tokens=ex.tokens, activations=ex.activations)
                for ex in f.activation_examples
            ],
            max_activation=f.max_activation,
        )
        for f in features
    ]
