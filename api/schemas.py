"""Pydantic request/response models for the Jimterpretability API."""

from typing import Literal
from pydantic import BaseModel, Field


# ── Sessions ──────────────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    model_name: Literal["gpt2-small", "gemma-2-2b", "llama-3.2-1b"] = "gpt2-small"
    layer: int = Field(default=8, ge=0, le=31)
    hook_type: Literal["resid_pre", "resid_post", "mlp_out"] = "resid_post"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"


class SessionResponse(BaseModel):
    session_id: str
    model_name: str
    layer: int
    hook_point: str
    device: str
    dtype: str
    neuronpedia_model_id: str | None
    neuronpedia_layer_id: str | None


class SessionSummary(BaseModel):
    session_id: str
    model_name: str
    layer: int
    hook_point: str
    device: str


# ── Concept Discovery ──────────────────────────────────────────────────────────

class ConceptDiscoverRequest(BaseModel):
    concept_label: str
    prompts: list[str] = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=100)
    activation_threshold: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=16, ge=1, le=64)
    aggregate_seq: Literal["mean", "max", "last"] = "mean"
    include_neuronpedia: bool = True


class FeatureStatResponse(BaseModel):
    feature_idx: int
    frequency: float
    mean_magnitude: float
    max_magnitude: float
    active_mean: float
    neuronpedia_label: str | None = None


class ConceptDiscoverResponse(BaseModel):
    concept_label: str
    n_prompts: int
    session_id: str
    top_features: list[FeatureStatResponse]


# ── Feature Inspection ─────────────────────────────────────────────────────────

class ActivationExampleResponse(BaseModel):
    tokens: list[str]
    activations: list[float]


class FeatureDetailResponse(BaseModel):
    feature_idx: int
    neuronpedia_label: str | None
    description: str | None
    activation_examples: list[ActivationExampleResponse]
    max_activation: float | None


class FeatureBatchRequest(BaseModel):
    feature_idxs: list[int] = Field(min_length=1, max_length=100)


# ── Interventions ──────────────────────────────────────────────────────────────

class SteerRequest(BaseModel):
    feature_idx: int = Field(ge=0)
    alpha: float = Field(description="Positive=amplify, negative=suppress")
    prompts: list[str] = Field(min_length=1, max_length=20)
    max_new_tokens: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)


class SteerResponse(BaseModel):
    feature_idx: int
    alpha: float
    steered_outputs: list[str]
    baseline_outputs: list[str]


class InterceptRequest(BaseModel):
    feature_idx: int = Field(ge=0)
    threshold: float = Field(gt=0.0)
    intercept_message: str
    prompt: str
    max_new_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)


class InterceptResponse(BaseModel):
    output: str
    was_intercepted: bool
    feature_idx: int
    threshold: float


class WeightEditRequest(BaseModel):
    feature_idx: int = Field(ge=0)
    layer: int = Field(ge=0)
    scale: float = Field(default=1.0, ge=0.0, le=1.0,
                         description="1.0 = fully project out, 0.0 = no change")


class WeightEditResponse(BaseModel):
    success: bool
    edit_id: str
    layer: int
    feature_idx: int
    scale: float


class WeightEditRestoreResponse(BaseModel):
    success: bool
    edit_id: str
    message: str
