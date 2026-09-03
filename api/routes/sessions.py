"""Session management routes: create, list, delete."""

from fastapi import APIRouter, HTTPException, Request

from jimterpretability.session import JimSession, SessionConfig
from api.schemas import SessionCreateRequest, SessionResponse, SessionSummary

router = APIRouter(tags=["sessions"])


def _get_registry(request: Request):
    return request.app.state.registry


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(body: SessionCreateRequest, request: Request):
    """Load a model + SAE into memory. Returns a session_id for subsequent calls."""
    registry = _get_registry(request)
    config = SessionConfig.build(
        model_name=body.model_name,
        layer=body.layer,
        hook_type=body.hook_type,
        device=body.device,
        dtype=body.dtype,
    )
    try:
        session = JimSession.create(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model/SAE: {exc}") from exc

    registry.add(session)
    return SessionResponse(**session.summary())


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(request: Request):
    return _get_registry(request).list_sessions()


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    """Teardown session and free memory."""
    registry = _get_registry(request)
    try:
        registry.remove(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
