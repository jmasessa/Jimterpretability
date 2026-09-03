"""Output interception: check SAE feature activation at each generated token."""

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE


def generate_with_interception(
    model: HookedTransformer,
    sae: SAE,
    hook_point: str,
    feature_idx: int,
    threshold: float,
    intercept_message: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> tuple[str, bool]:
    """
    Generate text token-by-token. At each step, if the target SAE feature exceeds
    `threshold`, stop immediately and return `intercept_message` instead.

    Returns:
        (output_text, was_intercepted)

    Design: manual generation loop using model.run_with_hooks() per step rather than
    model.generate(), giving a clean Python-level intercept point after each forward pass.
    """
    tokens = model.to_tokens([prompt], prepend_bos=True)  # (1, T_input)
    generated = tokens.clone()
    eos_id = model.tokenizer.eos_token_id

    for _step in range(max_new_tokens):
        captured: dict[str, float | None] = {"feature_act": None}

        def capture_hook(value: Tensor, hook, _captured=captured) -> Tensor:
            last_act = value[0, -1, :].unsqueeze(0).to(sae.W_enc.dtype)  # (1, d_model)
            with torch.no_grad():
                features = sae.encode(last_act)  # (1, n_features)
            _captured["feature_act"] = features[0, feature_idx].item()
            return value

        with torch.no_grad():
            logits = model.run_with_hooks(
                generated,
                fwd_hooks=[(hook_point, capture_hook)],
                return_type="logits",
            )  # (1, T_current, vocab_size)

        feat_val = captured["feature_act"]
        if feat_val is not None and feat_val > threshold:
            return intercept_message, True

        next_logits = logits[0, -1, :]  # (vocab_size,)
        if temperature == 0.0 or temperature == 1.0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    n_input = tokens.shape[1]
    output_tokens = generated[0, n_input:]
    return model.to_string(output_tokens), False
