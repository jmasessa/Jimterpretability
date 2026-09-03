"""API integration tests using FastAPI TestClient with mocked sessions."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from jimterpretability.session import JimSession, SessionConfig


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_session_config():
    return SessionConfig(
        model_name="gpt2-small",
        layer=8,
        hook_point="blocks.8.hook_resid_post",
        device="cpu",
        dtype="float32",
    )


def make_mock_session(session_id="abc-123"):
    session = MagicMock(spec=JimSession)
    session.session_id = session_id
    session.config = SessionConfig(
        model_name="gpt2-small",
        layer=8,
        hook_point="blocks.8.hook_resid_post",
        device="cpu",
        dtype="float32",
    )
    session.neuronpedia_model_id = "gpt2-sm"
    session.neuronpedia_layer_id = "8-res-jb"
    session._weight_edit_backups = {}
    session.summary.return_value = {
        "session_id": session_id,
        "model_name": "gpt2-small",
        "layer": 8,
        "hook_point": "blocks.8.hook_resid_post",
        "device": "cpu",
        "dtype": "float32",
        "neuronpedia_model_id": "gpt2-sm",
        "neuronpedia_layer_id": "8-res-jb",
    }
    return session


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_create_session(client):
    mock_session = make_mock_session()
    with patch("jimterpretability.session.JimSession.create", return_value=mock_session):
        r = client.post("/api/v1/sessions", json={
            "model_name": "gpt2-small",
            "layer": 8,
            "hook_type": "resid_post",
        })
    assert r.status_code == 201
    data = r.json()
    assert "session_id" in data
    assert data["model_name"] == "gpt2-small"


def test_get_sessions_empty(client):
    r = client.get("/api/v1/sessions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_delete_missing_session(client):
    r = client.delete("/api/v1/sessions/does-not-exist")
    assert r.status_code == 404


def test_create_and_delete_session(client):
    mock_session = make_mock_session("del-test")
    with patch("jimterpretability.session.JimSession.create", return_value=mock_session):
        r = client.post("/api/v1/sessions", json={"model_name": "gpt2-small", "layer": 8})
    assert r.status_code == 201
    sid = r.json()["session_id"]

    r2 = client.delete(f"/api/v1/sessions/{sid}")
    assert r2.status_code == 204


def test_concept_discover_unknown_session(client):
    r = client.post("/api/v1/sessions/no-such-session/concepts/discover", json={
        "concept_label": "dogs",
        "prompts": ["dogs are great"],
    })
    assert r.status_code == 404


def test_steer_unknown_session(client):
    r = client.post("/api/v1/sessions/no-such-session/interventions/steer", json={
        "feature_idx": 0,
        "alpha": 5.0,
        "prompts": ["hello"],
    })
    assert r.status_code == 404
