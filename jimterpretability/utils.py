"""Device detection, model config tables, and Neuronpedia ID mappings."""

import torch

# Supported model configurations
MODEL_CONFIGS: dict[str, dict] = {
    "gpt2-small": {
        "tl_name": "gpt2",
        "sae_release": "gpt2-small-res-jb",
        "default_hook_type": "resid_pre",   # gpt2-small-res-jb uses resid_pre
        "n_layers": 12,
        "d_model": 768,
        "neuronpedia_model_id": "gpt2-small",
        "neuronpedia_layer_template": "{layer}-res-jb",
    },
    "gemma-2-2b": {
        "tl_name": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res",
        "default_hook_type": "resid_post",
        "n_layers": 26,
        "d_model": 2304,
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_layer_template": "{layer}-gemmascope-res-16k",
    },
    "llama-3.2-1b": {
        "tl_name": "meta-llama/Llama-3.2-1B",
        "sae_release": "llama_3_1b_r4_14k",
        "default_hook_type": "resid_post",
        "n_layers": 16,
        "d_model": 2048,
        "neuronpedia_model_id": None,
        "neuronpedia_layer_template": None,
    },
}

# SAE hook point templates per hook type
HOOK_POINT_TEMPLATES = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
}


def resolve_hook_point(hook_type: str, layer: int) -> str:
    template = HOOK_POINT_TEMPLATES.get(hook_type)
    if template is None:
        raise ValueError(f"Unknown hook_type '{hook_type}'. Choose from: {list(HOOK_POINT_TEMPLATES)}")
    return template.format(layer=layer)


def resolve_neuronpedia_ids(model_name: str, layer: int) -> tuple[str | None, str | None]:
    """Return (neuronpedia_model_id, neuronpedia_layer_id) for a given model + layer."""
    cfg = MODEL_CONFIGS[model_name]
    np_model_id = cfg["neuronpedia_model_id"]
    template = cfg["neuronpedia_layer_template"]
    np_layer_id = template.format(layer=layer) if template else None
    return np_model_id, np_layer_id


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batch_list(items: list, batch_size: int):
    """Yield successive batches from items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
