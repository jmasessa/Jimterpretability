"""PolicyLayer: composable, serializable collection of inference-time interventions.

Rules are never applied permanently — weight edits are applied before generation
and always restored in a finally block, so the model is never durably changed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from sae_lens import SAE


@dataclass
class PolicyLayer:
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rules: list[dict] = field(default_factory=list)

    # ── Rule helpers ──────────────────────────────────────────────────────────

    def add_steer(self, feature_idx: int, alpha: float) -> None:
        self.rules.append({"type": "steer", "feature_idx": feature_idx, "alpha": alpha})

    def add_block(self, feature_idx: int, threshold: float, message: str) -> None:
        self.rules.append({"type": "block", "feature_idx": feature_idx,
                            "threshold": threshold, "message": message})

    def add_weight_edit(self, feature_idx: int, layer: int, scale: float = 1.0) -> None:
        self.rules.append({"type": "weight_edit", "feature_idx": feature_idx,
                            "layer": layer, "scale": scale})

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, policies_dir: Path) -> Path:
        policies_dir.mkdir(parents=True, exist_ok=True)
        path = policies_dir / f"{self.id}.json"
        path.write_text(json.dumps({"name": self.name, "id": self.id, "rules": self.rules}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> PolicyLayer:
        data = json.loads(path.read_text())
        return cls(name=data["name"], id=data.get("id", str(uuid.uuid4())), rules=data.get("rules", []))

    # ── Generation ────────────────────────────────────────────────────────────

    def apply_generation(
        self,
        model: HookedTransformer,
        sae: SAE,
        hook_point: str,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        """
        Run generation with all policy rules active. Model weights are restored
        after generation even if an exception occurs.
        """
        steer_rules = [r for r in self.rules if r["type"] == "steer"]
        block_rules  = [r for r in self.rules if r["type"] == "block"]
        we_rules     = [r for r in self.rules if r["type"] == "weight_edit"]

        # Apply weight edits temporarily
        backups: list[tuple[int, dict]] = []
        for r in we_rules:
            from jimterpretability.interventions.weight_edit import (
                apply_rank1_weight_edit, compute_feature_direction,
            )
            direction = compute_feature_direction(sae, r["feature_idx"])
            backup = apply_rank1_weight_edit(model, r["layer"], direction, r["scale"])
            backups.append((r["layer"], backup))

        try:
            return self._generate(model, sae, hook_point, prompt,
                                  steer_rules, block_rules, max_new_tokens, temperature)
        finally:
            from jimterpretability.interventions.weight_edit import restore_weights
            for layer, backup in backups:
                restore_weights(model, layer, backup)

    def _generate(
        self,
        model: HookedTransformer,
        sae: SAE,
        hook_point: str,
        prompt: str,
        steer_rules: list[dict],
        block_rules: list[dict],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Token-by-token loop: apply steering via hook, check block rules per step."""
        # Pre-fetch steering vectors (decoder columns)
        steering = [(sae.W_dec[r["feature_idx"]].detach().clone(), r["alpha"]) for r in steer_rules]

        tokens = model.to_tokens([prompt], prepend_bos=True)  # (1, T)
        generated = tokens.clone()
        eos_id = model.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            captured: dict[str, Tensor | None] = {"feats": None}

            def hook_fn(value: Tensor, hook,
                        _sv=steering, _cap=captured, _sae=sae) -> Tensor:
                for direction, alpha in _sv:
                    value = value + alpha * direction.to(value.device, value.dtype)
                last = value[0, -1, :].unsqueeze(0).to(_sae.W_enc.dtype)
                with torch.no_grad():
                    _cap["feats"] = _sae.encode(last)[0]
                return value

            with torch.no_grad():
                logits = model.run_with_hooks(
                    generated,
                    fwd_hooks=[(hook_point, hook_fn)],
                    return_type="logits",
                )  # (1, T_cur, vocab)

            # Check block rules against steered activations
            feats = captured["feats"]
            if feats is not None:
                for r in block_rules:
                    if feats[r["feature_idx"]].item() > r["threshold"]:
                        return r["message"]

            next_logits = logits[0, -1, :]
            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if eos_id is not None and next_token.item() == eos_id:
                break

        n_input = tokens.shape[1]
        return model.to_string(generated[0, n_input:])
