"""JimSession: stateful container for a loaded model + SAE pair."""

import threading
import uuid
from dataclasses import dataclass, field

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

from jimterpretability.utils import MODEL_CONFIGS, resolve_hook_point, resolve_neuronpedia_ids


@dataclass
class SessionConfig:
    model_name: str
    layer: int
    hook_point: str
    device: str = "cpu"
    dtype: str = "float32"

    @classmethod
    def build(cls, model_name: str, layer: int, hook_type: str = "resid_post",
              device: str = "cpu", dtype: str = "float32") -> "SessionConfig":
        hook_point = resolve_hook_point(hook_type, layer)
        return cls(model_name=model_name, layer=layer, hook_point=hook_point,
                   device=device, dtype=dtype)


@dataclass
class JimSession:
    session_id: str
    config: SessionConfig
    model: HookedTransformer
    sae: SAE
    neuronpedia_model_id: str | None
    neuronpedia_layer_id: str | None
    _weight_edit_backups: dict = field(default_factory=dict)  # edit_id -> {"layer": int, "W_out": Tensor}

    @classmethod
    def create(cls, config: SessionConfig) -> "JimSession":
        """Load model and SAE. May take 30-120 seconds on first call."""
        if config.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{config.model_name}'. Choose from: {list(MODEL_CONFIGS)}")

        tl_name = MODEL_CONFIGS[config.model_name]["tl_name"]
        sae_release = MODEL_CONFIGS[config.model_name]["sae_release"]
        torch_dtype = getattr(torch, config.dtype)
        device = torch.device(config.device)

        model = HookedTransformer.from_pretrained(tl_name, dtype=torch_dtype)
        model = model.to(device)
        model.eval()

        sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=config.hook_point,
            device=str(device),
        )
        # Align dtypes
        sae = sae.to(torch_dtype)
        sae.eval()

        np_model_id, np_layer_id = resolve_neuronpedia_ids(config.model_name, config.layer)

        return cls(
            session_id=str(uuid.uuid4()),
            config=config,
            model=model,
            sae=sae,
            neuronpedia_model_id=np_model_id,
            neuronpedia_layer_id=np_layer_id,
        )

    def teardown(self) -> None:
        del self.model
        del self.sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "model_name": self.config.model_name,
            "layer": self.config.layer,
            "hook_point": self.config.hook_point,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "neuronpedia_model_id": self.neuronpedia_model_id,
            "neuronpedia_layer_id": self.neuronpedia_layer_id,
        }


class SessionRegistry:
    """Thread-safe in-process store for loaded JimSessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, JimSession] = {}
        self._lock = threading.RLock()

    def add(self, session: JimSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> JimSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' not found")
            return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' not found")
            session = self._sessions.pop(session_id)
        session.teardown()

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [s.summary() for s in self._sessions.values()]
