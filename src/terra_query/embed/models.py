"""Model registry + loader for the aerial-image branch of the embedding pipeline.

One production model lives here. Additional candidates can be registered
without code surgery: drop a new `ModelSpec` into `MODELS`, optionally bump
`PRODUCTION_MODEL_ID` once the candidate is validated, and the rest of the
pipeline (fetch_weights, embed_chips, run_n0_retrieval) picks it up.

The registry is data; `load()` is the only behavior in this module.

Current production model: GeoRSCLIP ViT-L/14-336 on RGB. Chosen over the
224 variant because the denser 24x24 patch grid (vs 16x16 at 224) gives
finer spatial attention for small features; the per-chip cost is ~2x but
per-query and storage costs are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MPS_BATCH_SIZE = 16  # safe baseline on a 32 GB M-series unified memory


@dataclass(frozen=True)
class ModelSpec:
    """Everything we need to fetch + load one CLIP-family model."""

    model_id: str  # canonical id used in filenames + manifest
    arch: str  # open_clip arch string, e.g. "ViT-L-14-336"
    source: str  # "huggingface" or "open_clip_pretrained"
    hf_repo: str | None  # e.g. "Zilun/GeoRSCLIP", or None for open_clip_pretrained
    hf_filename: str | None  # e.g. "ckpt/RS5M_ViT-L-14-336.pt", or None
    pretrained: str | None  # e.g. "openai" (open_clip cache key), or None
    image_size: int  # input resolution per model release
    embed_dim: int  # joint text/image embedding dim
    mps_batch_size: int = DEFAULT_MPS_BATCH_SIZE  # per-model batch ceiling on MPS


# the production model (and any candidates an experimenter registers later)
MODELS: dict[str, ModelSpec] = {
    "georsclip-vit-l-14-336": ModelSpec(
        # production model for the aerial branch.
        # GeoRSCLIP trained on RS5M; 336x336 input means a 24x24 patch grid
        # (vs 16x16 at 224) which gives finer spatial attention for small
        # features. Slower per forward pass (~2x vs 224).
        model_id="georsclip-vit-l-14-336",
        arch="ViT-L-14-336",
        source="huggingface",
        hf_repo="Zilun/GeoRSCLIP",
        hf_filename="ckpt/RS5M_ViT-L-14-336.pt",
        pretrained=None,
        image_size=336,
        embed_dim=768,
        # 16 keeps headroom on 32 GB unified memory while staying GPU-saturated;
        # the 336 variant has ~2.25x more transformer tokens than ViT-L/14 (224).
        mps_batch_size=16,
    ),
}

# PRODUCTION_MODEL_ID is the registered default for callers that DON'T
# load an experiment YAML (e.g. the in-memory `evaluate_concept` test
# fixture). Every R1+ CLI reads `model_id` from the experiment YAML;
# the YAML supersedes this constant for everything else. To swap the
# production model end-to-end (fetch + embed + load + regenerate + purge
# old) use `python -m terra_query.embed.cli.swap_model`.
PRODUCTION_MODEL_ID = "georsclip-vit-l-14-336"


def model_ids() -> list[str]:
    """Default model ids the pipeline iterates over (just the production model).

    For a one-off A/B against a candidate, register the candidate in
    `MODELS` and pass `--models <candidate_id>` to the embed / n0_retrieval
    CLIs; the default sweep stays clean.
    """
    return [PRODUCTION_MODEL_ID]


def spec(model_id: str) -> ModelSpec:
    if model_id not in MODELS:
        raise KeyError(f"unknown model_id {model_id!r}; known: {sorted(MODELS)}")
    return MODELS[model_id]


def _unwrap_state_dict(obj):
    """Some checkpoints wrap the state_dict under 'state_dict' / 'model'
    and may include optimizer state. Find the actual state dict."""
    if not isinstance(obj, dict):
        return obj
    for key in ("state_dict", "model", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def _strip_ddp_prefix(sd: dict) -> dict:
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def load(model_id: str, weights_path: str | Path | None, device: str):
    """Load model + image preprocess + text tokenizer onto `device`.

    For `open_clip_pretrained`, `weights_path` is ignored (open_clip handles
    its own cache via `pretrained=<tag>`). For `huggingface`, `weights_path`
    must point to the local .pt file fetched by `fetch_weights`.

    Returns: (model, preprocess, tokenizer). Model is in eval() mode on
    `device`.
    """
    import open_clip
    import torch

    s = spec(model_id)

    if s.source == "open_clip_pretrained":
        model, _, preprocess = open_clip.create_model_and_transforms(
            s.arch, pretrained=s.pretrained, device=device
        )
    elif s.source == "huggingface":
        if weights_path is None:
            raise ValueError(f"{model_id}: huggingface source requires weights_path")
        model, _, preprocess = open_clip.create_model_and_transforms(
            s.arch, pretrained=None, device=device
        )
        raw = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        sd = _strip_ddp_prefix(_unwrap_state_dict(raw))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(
                f"{model_id}: state dict missing {len(missing)} keys, first 5: {missing[:5]}"
            )
        if unexpected:
            print(
                f"[load {model_id}] ignored {len(unexpected)} unexpected keys "
                f"(optimizer / training state), first 3: {unexpected[:3]}"
            )
    else:
        raise ValueError(f"unknown source {s.source!r} for {model_id}")

    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(s.arch)
    return model, preprocess, tokenizer


def load_production(device: str):
    """Load the production model on `device`, pulling weights_path from manifest.

    Convenience wrapper around `load()` for test fixtures and any consumer
    that just wants "the current production model" without threading the
    manifest lookup themselves.
    """
    import json

    from terra_query.core.paths import MODEL_WEIGHTS_MANIFEST

    manifest = json.loads(MODEL_WEIGHTS_MANIFEST.read_text())
    entry = manifest["models"][PRODUCTION_MODEL_ID]
    return load(PRODUCTION_MODEL_ID, weights_path=entry.get("weights_path"), device=device)
