"""Gradio UI for Jimterpretability."""

from __future__ import annotations

import asyncio
import os
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

def _list_policies() -> list[tuple[str, str]]:
    """Return (human name, filename) tuples for all saved rulebooks."""
    entries = []
    for p in sorted(POLICIES_DIR.glob("*.json")):
        try:
            pl = PolicyLayer.load(p)
            entries.append((pl.name, p.name))
        except Exception:
            entries.append((p.stem, p.name))  # fall back to filename stem
    return entries


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
    cfg = MODEL_CONFIGS.get(model_name, {})
    hook = cfg.get("default_hook_type", "resid_pre")
    return gr.update(value=hook)


def cb_unload_model():
    global _session
    if _session is None:
        return "No model is currently loaded."
    name = _session.config.model_name
    _session.teardown()
    _session = None
    return f"✅  {name} unloaded — memory freed."


def cb_load_model(model_name, layer, hook_type, device, dtype, hf_token):
    global _session
    if _session is not None:
        try:
            _session.teardown()
        except Exception:
            pass
        _session = None

    # Set the HuggingFace token in the environment for the duration of this
    # call only. It is never written to disk or stored by Jimterpretability.
    _prev_token = os.environ.get("HF_TOKEN")
    token = (hf_token or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token

    try:
        config = SessionConfig.build(
            model_name=model_name, layer=int(layer),
            hook_type=hook_type, device=device, dtype=dtype,
        )
        _session = JimSession.create(config)
        n = _session.sae.W_dec.shape[0]
        return (
            f"✅  Ready!  Model: {model_name}  ·  Reading layer {int(layer)}  ·  {n:,} concepts available to inspect",
            gr.update(value=int(layer)),
        )
    except Exception as e:
        return f"❌  {e}", gr.update()
    finally:
        # Always restore the environment — token never lingers
        if _prev_token is not None:
            os.environ["HF_TOKEN"] = _prev_token
        elif "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]


async def cb_discover(concept, prompts_raw, top_k, threshold, agg_mode, include_np, feat_state):
    global _session
    if _session is None:
        return None, "❌  No model loaded — go to the Setup tab first.", feat_state

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

        df = pd.DataFrame(rows, columns=["Feature ID", "Hit Rate", "Avg Strength", "Peak Strength", "What it encodes (Neuronpedia)"])
        status = f"✅  Found top {len(stats)} internal concepts for '{concept}' across {len(prompts)} prompts.  Click any row to select a concept."
        return df, status, new_state

    except Exception as e:
        return None, f"❌  {e}", feat_state


def cb_select_feature(evt: gr.SelectData, feat_state):
    if not feat_state or evt.index[0] >= len(feat_state):
        return gr.update(), gr.update(), gr.update()
    f = feat_state[evt.index[0]]
    label = f.get("label", "—")
    badge = f"**Selected → Concept #{f['feature_idx']}** — {label}"
    return f["feature_idx"], badge, badge


def cb_toggle_panels(intervention_type):
    return (
        gr.update(visible=intervention_type == "Steer"),
        gr.update(visible=intervention_type == "Block"),
        gr.update(visible=intervention_type in ("Weight Edit", "Suppress")),
    )


def cb_add_rule(feat_idx, intervention, alpha, blk_thresh, blk_msg, we_layer, we_scale, policy_state):
    if feat_idx is None or str(feat_idx).strip() == "":
        return policy_state, _rules_to_df(policy_state["rules"]), "❌  No concept selected — click a row in the Discover tab first."
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
    return new_state, _rules_to_df(rules), f"✅  Added {intervention} rule for concept #{feat_idx} ({detail})"


def cb_delete_rule(rule_idx, policy_state):
    try:
        idx = int(rule_idx)
        rules = list(policy_state.get("rules", []))
        if 0 <= idx < len(rules):
            rules.pop(idx)
            new_state = {**policy_state, "rules": rules}
            return new_state, _rules_to_df(rules), f"✅  Deleted rule #{idx}"
        return policy_state, _rules_to_df(policy_state["rules"]), "❌  Invalid rule number."
    except (ValueError, TypeError):
        return policy_state, _rules_to_df(policy_state["rules"]), "❌  Enter a valid rule number."


def cb_save_policy(name, policy_state):
    pl = PolicyLayer(name=name, rules=policy_state.get("rules", []))
    path = pl.save(POLICIES_DIR)
    choices = _list_policies()
    return (
        {**policy_state, "name": name, "id": pl.id},
        gr.update(choices=choices),
        gr.update(choices=[("None (no rulebook)", "none")] + choices),
        f"✅  Saved rulebook '{name}'",
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

    if policy_file and policy_file not in ("none", "None (no rulebook)"):
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
            with_policy = "(No rules in rulebook — output is identical to baseline)"

        return baseline, with_policy
    except Exception as e:
        return f"❌  {e}", f"❌  {e}"


# ── UI Layout ──────────────────────────────────────────────────────────────────

CSS = """
.status-box textarea { font-family: monospace; font-size: 13px; }
.feature-badge { font-size: 15px; padding: 8px 12px; background: #eef2ff; border-left: 4px solid #6366f1; border-radius: 4px; margin-top: 8px; }
.tip-box { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px; font-size: 14px; }
.warn-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 8px 12px; border-radius: 4px; font-size: 14px; }
footer { display: none !important; }
"""

with gr.Blocks(title="Jimterpretability") as demo:

    feat_state   = gr.State([])
    policy_state = gr.State({"name": "New Rulebook", "id": "", "rules": []})

    gr.Markdown("""
# 🔬 Jimterpretability

**What is this?**
Language models like GPT secretly track thousands of internal "concepts" as they generate text — things like *dogs*, *legal language*, *negative sentiment*, or *Staffordshire Terriers*.
This tool lets you find those hidden concepts, see which ones fire when you feed the model specific prompts, and then control them: amplify, suppress, or intercept them entirely.

**Your workflow:** &nbsp; ⚙️ Setup → 🔍 Discover → 📋 Build Rulebook → ✨ Generate
""")

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════
        # Tab 1 — Setup
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️  Setup"):
            gr.Markdown("""
### Step 1: Load a model
Choose an AI model and click **Load Model**. This prepares the model so we can peek inside it.
The first time you load a model it will download the weights from the internet (~500 MB for GPT-2 Small).
""")
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value="gpt2-small",
                        label="Which AI model?",
                        info="GPT-2 Small is the best starting point — it's small, fast, and well-studied. Larger models take longer to load and need more memory.",
                    )
                    layer_sl = gr.Slider(
                        0, 31, value=8, step=1,
                        label="Which layer to inspect?",
                        info="Think of the model as having many processing stages (layers), like floors in a building. Layer 8 of 12 is in the middle — a good default. Earlier layers handle basic patterns (grammar, word shapes); later layers handle meaning and reasoning.",
                    )
                    hook_dd = gr.Dropdown(
                        choices=["resid_pre", "resid_post", "mlp_out"],
                        value="resid_pre",
                        label="Where inside the layer to look?",
                        info="Each layer has a few internal checkpoints. 'resid_pre' reads the model's state just before this layer does its work. This is set automatically based on the model you choose.",
                    )
                    with gr.Row():
                        device_dd = gr.Dropdown(
                            ["cpu", "cuda", "mps"], value="cpu",
                            label="Hardware",
                            info="mps = Apple Silicon (M1/M2/M3 Mac), cuda = NVIDIA GPU, cpu = works everywhere but slower",
                        )
                        dtype_dd = gr.Dropdown(
                            ["float32", "bfloat16"], value="float32",
                            label="Precision",
                            info="float32 is more accurate. bfloat16 uses half the memory and is faster — useful for larger models.",
                        )
                    hf_token_tb = gr.Textbox(
                        label="HuggingFace Access Token  (only needed for gated models like Llama)",
                        placeholder="hf_••••••••••••••••••••••••••••••••",
                        type="password",
                        info="Required for models that need a licence agreement (e.g. Llama). GPT-2 and Gemma do not need this. Your token is used only for the download and is never saved to disk by this app.",
                    )
                    with gr.Row():
                        load_btn   = gr.Button("⚡  Load Model", variant="primary", size="lg")
                        unload_btn = gr.Button("🗑️  Unload Model", variant="stop", size="lg")

                with gr.Column(scale=2):
                    setup_status = gr.Textbox(
                        label="Status", interactive=False, lines=2,
                        value="No model loaded yet. Configure settings on the left and click Load Model.",
                        elem_classes=["status-box"],
                    )
                    with gr.Accordion("🔑 How to get a HuggingFace token (needed for Llama)", open=False):
                        gr.Markdown("""
Some models (like Meta's Llama) require you to agree to a licence before downloading.
Here's how to get access in two steps:

1. **Accept the licence** — visit [huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
   and click **Agree and access repository**. Approval is usually instant.

2. **Create a token** — go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   click **New token**, give it a name, select **Read** access, and copy the token (starts with `hf_`).

Paste the token into the field above before clicking Load Model.
Once the model has downloaded to your computer, you won't need the token again for that model.

**Privacy:** This app sets the token as a temporary environment variable only for the duration
of the download. It is deleted immediately after, and is never written to any file on disk.
""")
                    with gr.Accordion("💡 What is a 'layer' and why does it matter?", open=False):
                        gr.Markdown("""
A language model processes text through many sequential stages called **layers** — GPT-2 Small has 12.
Each layer refines the model's understanding, passing information to the next.

At each layer, a **Sparse Autoencoder (SAE)** acts like a translation dictionary:
it converts the model's raw internal numbers into ~16,000 named *features* (internal concepts).
Think of it as decoding the model's private shorthand into something humans can read.

**Why layer 8?** It's in the middle of GPT-2, where abstract meaning tends to live.
Layer 0 is still figuring out basic word shapes; layer 11 is already deciding what word to output next.
Layer 8 tends to hold the richest conceptual representations.
""")
                    with gr.Accordion("💡 Recommended settings for beginners", open=True):
                        gr.Markdown("""
| Setting | Best choice | Why |
|---|---|---|
| Model | **gpt2-small** | Small, fast, best-supported |
| Layer | **8** | Middle of the network — rich in meaning |
| Where to look | **resid_pre** | Set automatically for GPT-2 |
| Hardware | **mps** (Mac) or **cpu** | Use cuda if you have an NVIDIA GPU |
| Precision | **float32** | Most accurate, fine for GPT-2 Small |
""")

        # ═══════════════════════════════════════════════════════════════════
        # Tab 2 — Discover Features
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍  Discover Features"):
            gr.Markdown("""
### Step 2: Find which internal concepts fire for your topic
Write example sentences about a topic (e.g. Staffordshire Terriers), one per line.
The tool runs them through the model and finds which of the ~16,000 internal concepts activated most
consistently — those are the model's internal representation of your topic.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    concept_tb = gr.Textbox(
                        label="Topic name",
                        placeholder="e.g. Staffordshire Terriers",
                        info="Just a label for your own reference — it doesn't affect the analysis.",
                    )
                    prompts_ta = gr.Textbox(
                        label="Example sentences about your topic  (one per line)",
                        placeholder=(
                            "The Staffordshire Bull Terrier is a medium-sized breed.\n"
                            "American Staffordshire Terriers are loyal and protective dogs.\n"
                            "Staffies were originally bred for bull-baiting in England.\n"
                            "My Staffy loves playing fetch at the park.\n"
                            "Staffordshire Terriers have a muscular, stocky build.\n"
                            "..."
                        ),
                        lines=12,
                        info="Aim for 20–400 sentences. More variety = more reliable results. Include different ways the topic might come up naturally in text.",
                    )
                    with gr.Row():
                        top_k_sl = gr.Slider(
                            5, 50, value=20, step=1,
                            label="How many top concepts to show?",
                            info="Returns this many of the most frequently activated internal concepts. 20 is a good starting point.",
                        )
                        thresh_sl = gr.Slider(
                            0.0, 2.0, value=0.0, step=0.05,
                            label="Minimum activation strength",
                            info="Only count a concept as 'active' for a sentence if its strength exceeds this number. Leave at 0 to catch everything.",
                        )
                    agg_radio = gr.Radio(
                        ["mean", "max", "last"], value="mean",
                        label="How to measure activation across the sentence?",
                        info=(
                            "mean: average across all words (best for general topics)  |  "
                            "max: use the single highest spike (best for specific trigger words)  |  "
                            "last: only look at the final word"
                        ),
                    )
                    include_np_cb = gr.Checkbox(
                        value=True,
                        label="Look up human-readable concept names from Neuronpedia (requires internet)",
                        info="Neuronpedia is a public database where researchers have labelled what thousands of internal model concepts mean. This fetches those labels so you see 'dogs and canines' instead of just a number.",
                    )
                    discover_btn = gr.Button("🔍  Find Concepts", variant="primary", size="lg")

                with gr.Column(scale=2):
                    discover_status = gr.Textbox(
                        label="Status", interactive=False, lines=2,
                        elem_classes=["status-box"],
                    )
                    features_table = gr.Dataframe(
                        headers=["Feature ID", "Hit Rate", "Avg Strength", "Peak Strength", "What it encodes (Neuronpedia)"],
                        interactive=False,
                        label="Internal concepts ranked by how consistently they fired across your sentences",
                        wrap=True,
                    )
                    selected_feat_display = gr.Markdown(
                        "*Click any row above to select a concept and carry it into the Rulebook Builder →*",
                        elem_classes=["feature-badge"],
                    )
                    with gr.Accordion("💡 How to read this table", open=False):
                        gr.Markdown("""
| Column | What it means |
|---|---|
| **Feature ID** | The internal ID number for this concept inside the model. Think of it as the concept's serial number. |
| **Hit Rate** | What percentage of your sentences activated this concept. 85% means it fired in 85 out of 100 sentences — a strong signal it's related to your topic. |
| **Avg Strength** | The average intensity of activation across all your sentences (including ones where it didn't fire at all). Higher = more dominant concept. |
| **Peak Strength** | The strongest single activation seen across all sentences. Useful for spotting concepts that fire very strongly on specific trigger phrases. |
| **What it encodes** | A human-written label from the Neuronpedia database describing what this concept represents. A "—" means no label exists yet for this concept. |

**What to look for:** A concept strongly linked to your topic will have a high Hit Rate (>60%) *and* a descriptive Neuronpedia label that matches what you'd expect.
""")

        # ═══════════════════════════════════════════════════════════════════
        # Tab 3 — Policy Builder
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📋  Rulebook Builder"):
            gr.Markdown("""
### Step 3: Build a rulebook that controls how the model behaves
Select a concept from the Discover tab, choose what to do when the model "thinks" about it, and add it as a rule.
You can stack multiple rules into a **rulebook** and save it for reuse.
**Nothing here permanently changes the model** — rules are applied on the fly during generation and can be removed at any time.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 1 · Which concept do you want to control?")
                    feat_idx_nb = gr.Number(
                        label="Concept ID",
                        precision=0,
                        info="This is filled in automatically when you click a row in the Discover tab. You can also type a Feature ID number here directly.",
                    )
                    feat_label_md = gr.Markdown("*No concept selected — go to the Discover tab and click a row*", elem_classes=["feature-badge"])

                    gr.Markdown("#### 2 · What should happen when the model thinks about this concept?")
                    intervention_radio = gr.Radio(
                        ["Steer", "Block", "Weight Edit", "Suppress"],
                        value="Steer",
                        label="Choose an action",
                    )

                    # Steer panel
                    with gr.Group(visible=True) as steer_panel:
                        gr.Markdown("""
**🧭 Steer — nudge the model's thinking in a direction**

Imagine whispering in the model's ear while it writes. Steering adds (or subtracts) the strength of this concept at every step of generation.

- **Positive number** → the model thinks more about this concept (amplify)
- **Negative number** → the model avoids this concept (suppress)

*Example: Set alpha to −20 to make the model avoid mentioning Staffordshire Terriers even when asked about dog breeds.*
""", elem_classes=["tip-box"])
                        alpha_sl = gr.Slider(
                            -50, 50, value=10, step=0.5,
                            label="Strength & direction  (negative = push away, positive = pull toward)",
                            info="Start with ±10 and adjust. Values above ±30 can make output incoherent.",
                        )

                    # Block panel
                    with gr.Group(visible=False) as block_panel:
                        gr.Markdown("""
**🚫 Block — set a tripwire**

The model checks this concept's strength at every word it generates.
The moment it exceeds your threshold, generation stops immediately and returns your message instead — the model never finishes its sentence.

*Example: Set threshold to 0.5 and message to "Sorry, I can't discuss this topic." — the model will trip over the wire the instant it starts thinking about Staffordshire Terriers.*
""", elem_classes=["tip-box"])
                        blk_thresh_sl = gr.Slider(
                            0.0, 10.0, value=0.5, step=0.05,
                            label="Tripwire sensitivity  (lower = triggers more easily)",
                            info="If the concept's activation strength goes above this number during generation, the block fires. Start around 0.5 and tune up if it triggers too easily.",
                        )
                        blk_msg_tb = gr.Textbox(
                            label="Message to show instead",
                            value="I'm unable to discuss this topic.",
                            lines=2,
                            info="This is what the user will see when the tripwire fires.",
                        )

                    # Weight Edit / Suppress panel
                    with gr.Group(visible=False) as we_panel:
                        gr.Markdown("""
**🔇 Weight Edit / Suppress — temporarily mute the concept**

This directly edits the model's internal calculation to reduce how much influence this concept has.
Think of it as turning down the volume on a specific channel.

The model is automatically restored to its original state after each generation — **nothing is permanently changed**.

- **Weight Edit** lets you choose a partial mute (e.g. 50% reduction)
- **Suppress** is a full mute (100% reduction)

*Note: This is the most drastic option. Use Steer for subtle adjustments, Block for a hard stop.*
""", elem_classes=["warn-box"])
                        we_layer_nb = gr.Number(
                            label="Which layer to mute?",
                            value=8, precision=0,
                            info="Should match the layer you loaded in Setup. This is set automatically.",
                        )
                        we_scale_sl = gr.Slider(
                            0.0, 1.0, value=1.0, step=0.05,
                            label="How much to reduce it?  (0.0 = no change, 1.0 = fully silenced)",
                            info="1.0 removes the concept's influence entirely. 0.5 cuts it in half. Only visible for Weight Edit — Suppress always uses 1.0.",
                        )

                    add_rule_btn = gr.Button("➕  Add This Rule to Rulebook", variant="primary", size="lg")
                    rule_status = gr.Textbox(label="", interactive=False, lines=1, elem_classes=["status-box"])

                with gr.Column(scale=2):
                    gr.Markdown("#### Your current rulebook")
                    gr.Markdown(
                        "All rules below will be applied together when you generate text. "
                        "You can have multiple rules for different concepts.",
                        elem_classes=["tip-box"],
                    )
                    policy_name_tb = gr.Textbox(
                        label="Rulebook name",
                        value="New Rulebook",
                        info="Give your rulebook a descriptive name before saving.",
                    )
                    policy_table = gr.Dataframe(
                        headers=["#", "Type", "Feature", "Configuration"],
                        interactive=False,
                        label="Rules in this rulebook",
                        wrap=True,
                    )
                    with gr.Row():
                        delete_idx_nb = gr.Number(
                            label="Rule # to remove",
                            value=0, precision=0, minimum=0,
                            info="Enter the number from the # column above, then click Remove.",
                        )
                        delete_rule_btn = gr.Button("🗑️  Remove Rule", variant="stop")
                    rule_del_status = gr.Textbox(label="", interactive=False, lines=1, elem_classes=["status-box"])

                    gr.Markdown("---")
                    gr.Markdown("#### Save & load rulebooks")
                    gr.Markdown(
                        "Save your rulebook to reuse it later. Rulebooks are stored as simple files on your computer.",
                        elem_classes=["tip-box"],
                    )
                    save_policy_btn = gr.Button("💾  Save Rulebook", variant="primary")
                    with gr.Row():
                        saved_policies_dd = gr.Dropdown(
                            choices=_list_policies(),
                            label="Previously saved rulebooks",
                            allow_custom_value=False,
                            info="Select a saved rulebook from the list, then click Load. Shows the name you gave it when saving.",
                        )
                        load_policy_btn = gr.Button("📂  Load")
                    policy_io_status = gr.Textbox(label="", interactive=False, lines=1, elem_classes=["status-box"])

        # ═══════════════════════════════════════════════════════════════════
        # Tab 4 — Generate
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("✨  Generate"):
            gr.Markdown("""
### Step 4: See your rulebook in action
Type a prompt and generate text twice — once normally (baseline) and once with your rulebook applied.
Compare the two outputs side-by-side to see exactly what your rules changed.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    generate_policy_dd = gr.Dropdown(
                        choices=[("None (no rulebook)", "none")] + _list_policies(),
                        value="none",
                        label="Which rulebook to apply?",
                        info="Choose a saved rulebook, or select 'None' to generate without any rules (useful for comparing).",
                    )
                    gen_prompt_tb = gr.Textbox(
                        label="Your prompt",
                        placeholder="e.g. Tell me about Staffordshire Bull Terriers",
                        lines=5,
                        info="Type anything you'd like the model to respond to. Try something likely to trigger the concepts in your rulebook.",
                    )
                    with gr.Row():
                        max_tokens_sl = gr.Slider(
                            10, 300, value=100, step=10,
                            label="Maximum response length (words)",
                            info="How many words the model is allowed to generate before it stops. 100 is a good default.",
                        )
                        temperature_sl = gr.Slider(
                            0.0, 2.0, value=1.0, step=0.1,
                            label="Creativity / randomness",
                            info="0.0 = always picks the most likely next word (predictable). 1.0 = normal. 2.0 = highly random and creative (may become incoherent). Start at 1.0.",
                        )
                    generate_btn = gr.Button("🚀  Generate", variant="primary", size="lg")
                    with gr.Accordion("💡 Tips for a good comparison", open=False):
                        gr.Markdown("""
- Use a prompt that's **likely to trigger** the concept you're controlling.
  If your rulebook suppresses Staffordshire Terriers, try asking directly about them.
- The **Baseline** column shows what the model would say *without* any intervention.
- The **With Rulebook** column shows the effect of your rules.
- If the outputs look identical, your rule may not be triggering — try lowering the Block threshold or increasing the Steer alpha.
- Try the same prompt multiple times with different Temperature values to see how stability changes.
""")

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📄 Baseline\n*Unmodified model — no rules applied*")
                            baseline_out = gr.Textbox(label="", lines=12, interactive=False)
                        with gr.Column():
                            gr.Markdown("### 🛡️ With Rulebook\n*Model output after your rules are applied*")
                            policy_out = gr.Textbox(label="", lines=12, interactive=False)

    # ── Event wiring ─────────────────────────────────────────────────────────

    model_dd.change(cb_model_changed, inputs=[model_dd], outputs=[hook_dd])

    unload_btn.click(cb_unload_model, inputs=[], outputs=[setup_status])

    load_btn.click(
        cb_load_model,
        inputs=[model_dd, layer_sl, hook_dd, device_dd, dtype_dd, hf_token_tb],
        outputs=[setup_status, we_layer_nb],
    )

    discover_btn.click(
        cb_discover,
        inputs=[concept_tb, prompts_ta, top_k_sl, thresh_sl, agg_radio, include_np_cb, feat_state],
        outputs=[features_table, discover_status, feat_state],
    )

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
