"""Gradio UI for Jimterpretability."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from jimterpretability.session import JimSession, SessionConfig
from jimterpretability.activations import extract_activations
from jimterpretability.features import aggregate_features
from jimterpretability.neuronpedia import NeuronpediaClient
from jimterpretability.policy import PolicyLayer
from jimterpretability.interventions.steering import compute_baseline
from jimterpretability.utils import MODEL_CONFIGS

POLICIES_DIR = Path(__file__).parent.parent / "policies"
POLICIES_DIR.mkdir(exist_ok=True)

# The session holds large PyTorch tensors — keep it as a module-level global
# rather than gr.State to avoid Gradio trying to deep-copy it.
_session: JimSession | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _list_policies() -> list[str]:
    return sorted(p.name for p in POLICIES_DIR.glob("*.json"))


def _rules_to_df(rules: list[dict]) -> pd.DataFrame:
    if not rules:
        return pd.DataFrame(columns=["#", "Type", "Feature", "Configuration"])
    rows = []
    for i, r in enumerate(rules):
        if r["type"] == "steer":
            cfg = f"alpha = {r['alpha']:+.2f}  ({'amplify' if r['alpha'] > 0 else 'suppress'})"
        elif r["type"] == "block":
            msg = r["message"][:45] + ("…" if len(r["message"]) > 45 else "")
            cfg = f"threshold = {r['threshold']:.2f} | message: \"{msg}\""
        else:
            verb = "full suppression" if r["scale"] >= 1.0 else f"scale = {r['scale']:.2f}"
            cfg = f"layer {r['layer']} | {verb}"
        rows.append([i, r["type"].replace("_", " ").title(), r["feature_idx"], cfg])
    return pd.DataFrame(rows, columns=["#", "Type", "Feature", "Configuration"])


# ── Callbacks ──────────────────────────────────────────────────────────────────

def cb_model_changed(model_name):
    """Auto-select the correct hook type when the model dropdown changes."""
    cfg = MODEL_CONFIGS.get(model_name, {})
    hook = cfg.get("default_hook_type", "resid_pre")
    return gr.update(value=hook)


def cb_load_model(model_name, layer, hook_type, device, dtype):
    global _session
    if _session is not None:
        try:
            _session.teardown()
        except Exception:
            pass
        _session = None
    try:
        config = SessionConfig.build(
            model_name=model_name, layer=int(layer),
            hook_type=hook_type, device=device, dtype=dtype,
        )
        _session = JimSession.create(config)
        n = _session.sae.W_dec.shape[0]
        return (
            f"✅  {model_name}  ·  layer {int(layer)}  ·  {config.hook_point}  ·  {n:,} SAE features",
            gr.update(value=int(layer)),   # sync we_layer default to session layer
        )
    except Exception as e:
        return f"❌  {e}", gr.update()


async def cb_discover(concept, prompts_raw, top_k, threshold, agg_mode, include_np, feat_state):
    global _session
    if _session is None:
        return None, "❌  No model loaded — go to Setup first.", feat_state

    prompts = [p.strip() for p in prompts_raw.strip().splitlines() if p.strip()]
    if not prompts:
        return None, "❌  No prompts entered.", feat_state

    try:
        matrix = extract_activations(
            model=_session.model, sae=_session.sae,
            hook_point=_session.config.hook_point,
            prompts=prompts, batch_size=8,
            aggregate_seq=agg_mode, device=_session.config.device,
        )
        stats = aggregate_features(matrix, float(threshold), int(top_k))

        np_labels: dict[int, str] = {}
        if include_np and _session.neuronpedia_model_id and _session.neuronpedia_layer_id:
            async with NeuronpediaClient(
                model_id=_session.neuronpedia_model_id,
                layer_id=_session.neuronpedia_layer_id,
            ) as client:
                nf = await client.get_features_batch([s.feature_idx for s in stats])
                np_labels = {f.feature_idx: (f.label or "unlabeled") for f in nf}

        rows, new_state = [], []
        for s in stats:
            label = np_labels.get(s.feature_idx, "—")
            rows.append([
                s.feature_idx, f"{s.frequency:.1%}",
                f"{s.mean_magnitude:.4f}", f"{s.max_magnitude:.4f}", label,
            ])
            new_state.append({"feature_idx": s.feature_idx, "label": label})

        df = pd.DataFrame(rows, columns=["Feature", "Frequency", "Mean Mag", "Max Mag", "Neuronpedia Label"])
        status = f"✅  Top {len(stats)} features for '{concept}' across {len(prompts)} prompts — click a row to select it"
        return df, status, new_state

    except Exception as e:
        return None, f"❌  {e}", feat_state


def cb_select_feature(evt: gr.SelectData, feat_state):
    if not feat_state or evt.index[0] >= len(feat_state):
        return gr.update(), gr.update(), gr.update()
    f = feat_state[evt.index[0]]
    label = f.get("label", "—")
    badge = f"**Selected → Feature {f['feature_idx']}** — {label}"
    return f["feature_idx"], badge, badge


def cb_toggle_panels(intervention_type):
    return (
        gr.update(visible=intervention_type == "Steer"),
        gr.update(visible=intervention_type == "Block"),
        gr.update(visible=intervention_type in ("Weight Edit", "Suppress")),
    )


def cb_add_rule(feat_idx, intervention, alpha, blk_thresh, blk_msg, we_layer, we_scale, policy_state):
    if feat_idx is None or str(feat_idx).strip() == "":
        return policy_state, _rules_to_df(policy_state["rules"]), "❌  No feature selected."
    feat_idx = int(feat_idx)
    rules = list(policy_state.get("rules", []))

    if intervention == "Steer":
        rules.append({"type": "steer", "feature_idx": feat_idx, "alpha": float(alpha)})
        detail = f"alpha={float(alpha):+.2f}"
    elif intervention == "Block":
        rules.append({"type": "block", "feature_idx": feat_idx,
                      "threshold": float(blk_thresh), "message": blk_msg})
        detail = f"threshold={float(blk_thresh):.2f}"
    elif intervention == "Weight Edit":
        rules.append({"type": "weight_edit", "feature_idx": feat_idx,
                      "layer": int(we_layer), "scale": float(we_scale)})
        detail = f"layer={int(we_layer)} scale={float(we_scale):.2f}"
    else:  # Suppress
        layer = int(we_layer) if we_layer else (_session.config.layer if _session else 8)
        rules.append({"type": "weight_edit", "feature_idx": feat_idx, "layer": layer, "scale": 1.0})
        detail = f"layer={layer} (full)"

    new_state = {**policy_state, "rules": rules}
    return new_state, _rules_to_df(rules), f"✅  Added {intervention} rule for feature {feat_idx} ({detail})"


def cb_delete_rule(rule_idx, policy_state):
    try:
        idx = int(rule_idx)
        rules = list(policy_state.get("rules", []))
        if 0 <= idx < len(rules):
            rules.pop(idx)
            new_state = {**policy_state, "rules": rules}
            return new_state, _rules_to_df(rules), f"✅  Deleted rule #{idx}"
        return policy_state, _rules_to_df(policy_state["rules"]), "❌  Invalid rule index."
    except (ValueError, TypeError):
        return policy_state, _rules_to_df(policy_state["rules"]), "❌  Enter a valid rule number."


def cb_save_policy(name, policy_state):
    pl = PolicyLayer(name=name, rules=policy_state.get("rules", []))
    path = pl.save(POLICIES_DIR)
    choices = _list_policies()
    return (
        {**policy_state, "name": name, "id": pl.id},
        gr.update(choices=choices),
        gr.update(choices=["None (no policy)"] + choices),
        f"✅  Saved '{name}' → {path.name}",
    )


def cb_load_policy(filename, policy_state):
    if not filename:
        return policy_state, _rules_to_df(policy_state["rules"]), gr.update(), "❌  No file selected."
    path = POLICIES_DIR / filename
    if not path.exists():
        return policy_state, _rules_to_df(policy_state["rules"]), gr.update(), "❌  File not found."
    pl = PolicyLayer.load(path)
    new_state = {"name": pl.name, "id": pl.id, "rules": pl.rules}
    return new_state, _rules_to_df(pl.rules), gr.update(value=pl.name), f"✅  Loaded '{pl.name}' ({len(pl.rules)} rules)"


def cb_generate(prompt, policy_file, max_tokens, temperature, policy_state):
    global _session
    if _session is None:
        return "❌  No model loaded.", "❌  No model loaded."
    if not prompt.strip():
        return "❌  Empty prompt.", "❌  Empty prompt."

    if policy_file and policy_file != "None (no policy)":
        path = POLICIES_DIR / policy_file
        pl = PolicyLayer.load(path) if path.exists() else PolicyLayer(name="current", rules=policy_state.get("rules", []))
    else:
        pl = PolicyLayer(name="current", rules=policy_state.get("rules", []))

    try:
        baseline = compute_baseline(
            model=_session.model, prompts=[prompt],
            max_new_tokens=int(max_tokens), temperature=float(temperature),
        )[0]

        if pl.rules:
            with_policy = pl.apply_generation(
                model=_session.model, sae=_session.sae,
                hook_point=_session.config.hook_point,
                prompt=prompt, max_new_tokens=int(max_tokens),
                temperature=float(temperature),
            )
        else:
            with_policy = "(Policy has no rules — output is identical to baseline)"

        return baseline, with_policy
    except Exception as e:
        return f"❌  {e}", f"❌  {e}"


# ── UI Layout ──────────────────────────────────────────────────────────────────

CSS = """
.status-box textarea { font-family: monospace; font-size: 13px; }
.feature-badge { font-size: 15px; padding: 8px; background: #f0f4ff; border-radius: 6px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Jimterpretability") as demo:

    # ── Shared state ─────────────────────────────────────────────────────────
    feat_state   = gr.State([])
    policy_state = gr.State({"name": "New Policy", "id": "", "rules": []})

    # ── Header ───────────────────────────────────────────────────────────────
    gr.Markdown("""
# 🔬 Jimterpretability
*Discover, understand, and control features inside open-weight language models*

**Workflow:** Setup → Discover Features → Build Policy → Generate
""")

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════
        # Tab 1 — Setup
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️  Setup"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value="gpt2-small",
                        label="Model",
                        info="First load will download weights from HuggingFace (~500 MB for GPT-2 Small)",
                    )
                    layer_sl = gr.Slider(0, 31, value=8, step=1, label="SAE Layer",
                                         info="Which transformer layer's SAE to load")
                    hook_dd = gr.Dropdown(
                        choices=["resid_pre", "resid_post", "mlp_out"],
                        value="resid_pre", label="Hook Type",
                        info="Where in each block to attach the SAE — auto-updated when you change the model",
                    )
                    with gr.Row():
                        device_dd = gr.Dropdown(["cpu", "cuda", "mps"], value="cpu", label="Device")
                        dtype_dd  = gr.Dropdown(["float32", "bfloat16"], value="float32", label="dtype")
                    load_btn = gr.Button("Load Model", variant="primary", size="lg")

                with gr.Column(scale=2):
                    setup_status = gr.Textbox(
                        label="Status", interactive=False, lines=2,
                        value="No model loaded.",
                        elem_classes=["status-box"],
                    )
                    gr.Markdown("""
### Quick reference
| Setting | Recommended for starting |
|---|---|
| Model | **gpt2-small** |
| Layer | **8** (middle of the network, well-studied) |
| Hook type | **resid_post** (residual stream after each block) |
| Device | **mps** on Apple Silicon, **cuda** on NVIDIA, **cpu** otherwise |

Each layer has a pre-trained **Sparse Autoencoder (SAE)** that decomposes the dense residual stream activations into ~16,000 sparse, human-interpretable *features*.
When you feed prompts about a concept, we find which features light up most consistently — those are the model's internal representation of that concept.
""")

        # ═══════════════════════════════════════════════════════════════════
        # Tab 2 — Discover Features
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍  Discover Features"):
            with gr.Row():
                with gr.Column(scale=1):
                    concept_tb = gr.Textbox(
                        label="Concept label",
                        placeholder="e.g. Staffordshire Terriers",
                        info="Just a name for your reference — does not affect the computation",
                    )
                    prompts_ta = gr.Textbox(
                        label="Prompts  (one per line)",
                        placeholder=(
                            "The Staffordshire Bull Terrier is a medium-sized breed.\n"
                            "American Staffordshire Terriers are loyal and protective dogs.\n"
                            "Staffies were originally bred for bull-baiting in England.\n"
                            "My Staffy loves playing fetch at the park.\n"
                            "..."
                        ),
                        lines=12,
                        info="Use 20–400 prompts. More prompts → more reliable feature rankings.",
                    )
                    with gr.Row():
                        top_k_sl    = gr.Slider(5, 50, value=20, step=1, label="Top K features to return")
                        thresh_sl   = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Activation threshold")
                    agg_radio = gr.Radio(
                        ["mean", "max", "last"], value="mean",
                        label="Sequence aggregation",
                        info="How to combine activations across tokens in a prompt",
                    )
                    include_np_cb = gr.Checkbox(
                        value=True, label="Fetch Neuronpedia labels  (requires internet)"
                    )
                    discover_btn = gr.Button("🔍  Discover Features", variant="primary", size="lg")

                with gr.Column(scale=2):
                    discover_status = gr.Textbox(label="Status", interactive=False, lines=2,
                                                  elem_classes=["status-box"])
                    features_table = gr.Dataframe(
                        headers=["Feature", "Frequency", "Mean Mag", "Max Mag", "Neuronpedia Label"],
                        interactive=False,
                        label="Top Activated Features",
                        wrap=True,
                    )
                    selected_feat_display = gr.Markdown(
                        "*Click a row above to select a feature, then go to the Policy Builder tab*",
                        elem_classes=["feature-badge"],
                    )

        # ═══════════════════════════════════════════════════════════════════
        # Tab 3 — Policy Builder
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📋  Policy Builder"):
            with gr.Row():

                # Left: rule composer
                with gr.Column(scale=1):
                    gr.Markdown("### 1 · Select a feature")
                    feat_idx_nb  = gr.Number(
                        label="Feature index",
                        precision=0,
                        info="Auto-populated when you click a row in Discover Features, or enter manually",
                    )
                    feat_label_md = gr.Markdown("*No feature selected*", elem_classes=["feature-badge"])

                    gr.Markdown("### 2 · Choose an intervention")
                    intervention_radio = gr.Radio(
                        ["Steer", "Block", "Weight Edit", "Suppress"],
                        value="Steer",
                        label="Intervention type",
                    )

                    with gr.Group(visible=True) as steer_panel:
                        gr.Markdown("""
**Steer** — Adds a scaled copy of the feature's direction vector to the residual stream at every token.
- Positive alpha → amplifies the concept in the output
- Negative alpha → suppresses the concept
""")
                        alpha_sl = gr.Slider(-50, 50, value=10, step=0.5,
                                              label="Alpha  (negative = suppress, positive = amplify)")

                    with gr.Group(visible=False) as block_panel:
                        gr.Markdown("""
**Block** — Monitors the feature's activation at each generated token.
If it exceeds the threshold, generation stops immediately and your message is returned instead.
""")
                        blk_thresh_sl = gr.Slider(0.0, 10.0, value=0.5, step=0.05,
                                                    label="Activation threshold")
                        blk_msg_tb = gr.Textbox(
                            label="Intercept message",
                            value="I'm unable to discuss this topic.",
                            lines=2,
                        )

                    with gr.Group(visible=False) as we_panel:
                        gr.Markdown("""
**Weight Edit** — Temporarily removes the feature's direction from the MLP output matrix during generation.
The model is restored immediately after — this is **not permanent**.

**Suppress** sets scale = 1.0 (complete removal of the feature's influence).
""")
                        we_layer_nb = gr.Number(label="MLP layer to edit", value=8, precision=0)
                        we_scale_sl = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                                 label="Scale  (1.0 = full suppression)")

                    add_rule_btn = gr.Button("➕  Add Rule to Policy", variant="primary")
                    rule_status  = gr.Textbox(label="", interactive=False, lines=1,
                                               elem_classes=["status-box"])

                # Right: policy view
                with gr.Column(scale=2):
                    gr.Markdown("### Current Policy")
                    policy_name_tb = gr.Textbox(label="Policy name", value="New Policy")
                    policy_table = gr.Dataframe(
                        headers=["#", "Type", "Feature", "Configuration"],
                        interactive=False,
                        label="Rules  (to delete a rule, enter its # below and click Delete)",
                        wrap=True,
                    )
                    with gr.Row():
                        delete_idx_nb  = gr.Number(label="Rule # to delete", value=0, precision=0, minimum=0)
                        delete_rule_btn = gr.Button("🗑️  Delete Rule", variant="stop")
                    rule_del_status = gr.Textbox(label="", interactive=False, lines=1,
                                                  elem_classes=["status-box"])

                    gr.Markdown("---")
                    gr.Markdown("### Save / Load")
                    with gr.Row():
                        save_policy_btn = gr.Button("💾  Save Policy", variant="primary")
                    with gr.Row():
                        saved_policies_dd = gr.Dropdown(
                            choices=_list_policies(),
                            label="Saved policies",
                            allow_custom_value=False,
                        )
                        load_policy_btn = gr.Button("📂  Load")
                    policy_io_status = gr.Textbox(label="", interactive=False, lines=1,
                                                    elem_classes=["status-box"])

        # ═══════════════════════════════════════════════════════════════════
        # Tab 4 — Generate
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("✨  Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    generate_policy_dd = gr.Dropdown(
                        choices=["None (no policy)"] + _list_policies(),
                        value="None (no policy)",
                        label="Active policy",
                        info="'None' uses no rules. Select a saved policy, or leave this and use the rules from the Policy Builder tab.",
                    )
                    gen_prompt_tb = gr.Textbox(
                        label="Prompt",
                        placeholder="Tell me about Staffordshire Bull Terriers",
                        lines=5,
                    )
                    with gr.Row():
                        max_tokens_sl  = gr.Slider(10, 300, value=100, step=10, label="Max new tokens")
                        temperature_sl = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
                    generate_btn = gr.Button("🚀  Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Baseline  *(no policy)*")
                            baseline_out = gr.Textbox(label="", lines=12, interactive=False)
                        with gr.Column():
                            gr.Markdown("### With Policy")
                            policy_out = gr.Textbox(label="", lines=12, interactive=False)

    # ── Event wiring (all at the end so cross-tab refs resolve) ──────────────

    model_dd.change(
        cb_model_changed,
        inputs=[model_dd],
        outputs=[hook_dd],
    )

    load_btn.click(
        cb_load_model,
        inputs=[model_dd, layer_sl, hook_dd, device_dd, dtype_dd],
        outputs=[setup_status, we_layer_nb],
    )

    discover_btn.click(
        cb_discover,
        inputs=[concept_tb, prompts_ta, top_k_sl, thresh_sl, agg_radio, include_np_cb, feat_state],
        outputs=[features_table, discover_status, feat_state],
    )

    # Row click in Tab 2 → populate Tab 3 feature index + both displays
    features_table.select(
        cb_select_feature,
        inputs=[feat_state],
        outputs=[feat_idx_nb, feat_label_md, selected_feat_display],
    )

    intervention_radio.change(
        cb_toggle_panels,
        inputs=[intervention_radio],
        outputs=[steer_panel, block_panel, we_panel],
    )

    add_rule_btn.click(
        cb_add_rule,
        inputs=[feat_idx_nb, intervention_radio, alpha_sl, blk_thresh_sl,
                blk_msg_tb, we_layer_nb, we_scale_sl, policy_state],
        outputs=[policy_state, policy_table, rule_status],
    )

    delete_rule_btn.click(
        cb_delete_rule,
        inputs=[delete_idx_nb, policy_state],
        outputs=[policy_state, policy_table, rule_del_status],
    )

    save_policy_btn.click(
        cb_save_policy,
        inputs=[policy_name_tb, policy_state],
        outputs=[policy_state, saved_policies_dd, generate_policy_dd, policy_io_status],
    )

    load_policy_btn.click(
        cb_load_policy,
        inputs=[saved_policies_dd, policy_state],
        outputs=[policy_state, policy_table, policy_name_tb, policy_io_status],
    )

    generate_btn.click(
        cb_generate,
        inputs=[gen_prompt_tb, generate_policy_dd, max_tokens_sl, temperature_sl, policy_state],
        outputs=[baseline_out, policy_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CSS,
    )
