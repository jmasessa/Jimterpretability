"""Intervention routes: activation steering, output interception, weight editing."""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request

from jimterpretability.interventions.steering import steer, compute_baseline
from jimterpretability.interventions.interception import generate_with_interception
from jimterpretability.interventions.weight_edit import (
    apply_rank1_weight_edit,
    restore_weights,
    compute_feature_direction,
)
from api.schemas import (
    SteerRequest, SteerResponse,
    InterceptRequest, InterceptResponse,
    WeightEditRequest, WeightEditResponse,
    WeightEditRestoreResponse,
)

router = APIRouter(tags=["interventions"])


def _get_session(session_id: str, request: Request):
    try:
        return request.app.state.registry.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.post("/sessions/{session_id}/interventions/steer", response_model=SteerResponse)
async def steer_feature(session_id: str, body: SteerRequest, request: Request):
    """
    Generate text with a feature direction added to (or subtracted from) the residual stream.
    Returns both steered and baseline outputs for comparison.
    """
    session = _get_session(session_id, request)
    loop = asyncio.get_running_loop()

    steered, baseline = await asyncio.gather(
        loop.run_in_executor(
            None,
            lambda: steer(
                model=session.model,
                sae=session.sae,
                feature_idx=body.feature_idx,
                alpha=body.alpha,
                hook_point=session.config.hook_point,
                prompts=body.prompts,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
            ),
        ),
        loop.run_in_executor(
            None,
            lambda: compute_baseline(
                model=session.model,
                prompts=body.prompts,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
            ),
        ),
    )

    return SteerResponse(
        feature_idx=body.feature_idx,
        alpha=body.alpha,
        steered_outputs=steered,
        baseline_outputs=baseline,
    )


@router.post("/sessions/{session_id}/interventions/intercept", response_model=InterceptResponse)
async def intercept_feature(session_id: str, body: InterceptRequest, request: Request):
    """
    Generate text; if feature activation exceeds threshold at any token, return intercept_message.
    """
    session = _get_session(session_id, request)
    loop = asyncio.get_running_loop()

    output, was_intercepted = await loop.run_in_executor(
        None,
        lambda: generate_with_interception(
            model=session.model,
            sae=session.sae,
            hook_point=session.config.hook_point,
            feature_idx=body.feature_idx,
            threshold=body.threshold,
            intercept_message=body.intercept_message,
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
        ),
    )

    return InterceptResponse(
        output=output,
        was_intercepted=was_intercepted,
        feature_idx=body.feature_idx,
        threshold=body.threshold,
    )


@router.post("/sessions/{session_id}/interventions/weight-edit", response_model=WeightEditResponse)
async def weight_edit(session_id: str, body: WeightEditRequest, request: Request):
    """
    Permanently project a feature direction out of MLP W_out at the given layer.
    Stores a backup so the edit can be reverted via DELETE.
    """
    session = _get_session(session_id, request)

    feature_dir = compute_feature_direction(session.sae, body.feature_idx)
    backup = apply_rank1_weight_edit(
        model=session.model,
        layer=body.layer,
        feature_direction=feature_dir,
        scale=body.scale,
    )

    session._weight_edit_backups[backup["edit_id"]] = backup

    return WeightEditResponse(
        success=True,
        edit_id=backup["edit_id"],
        layer=body.layer,
        feature_idx=body.feature_idx,
        scale=body.scale,
    )


@router.delete(
    "/sessions/{session_id}/interventions/weight-edit/{edit_id}",
    response_model=WeightEditRestoreResponse,
)
async def restore_weight_edit(session_id: str, edit_id: str, request: Request):
    """Restore model weights from a previous weight edit backup."""
    session = _get_session(session_id, request)

    backup = session._weight_edit_backups.pop(edit_id, None)
    if backup is None:
        raise HTTPException(status_code=404, detail=f"Edit '{edit_id}' not found")

    restore_weights(model=session.model, layer=backup["layer"], backup=backup)

    return WeightEditRestoreResponse(
        success=True,
        edit_id=edit_id,
        message=f"Weights restored for layer {backup['layer']}",
    )
