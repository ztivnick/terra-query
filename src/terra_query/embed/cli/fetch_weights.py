"""CLI: download model weights, verify they load on MPS, write a manifest.

Idempotent: skips re-download if the destination file's SHA256 matches the
manifest entry. Always re-verifies the manifest entries against on-disk files.

Defaults to the experiment YAML's `model_id` if no `--models` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

from terra_query.core import config
from terra_query.core.paths import (
    MODEL_WEIGHTS_DIR,
    MODEL_WEIGHTS_MANIFEST,
    model_weights_dir,
)
from terra_query.embed import models


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _load_manifest() -> dict:
    if MODEL_WEIGHTS_MANIFEST.exists():
        return json.loads(MODEL_WEIGHTS_MANIFEST.read_text())
    return {"models": {}}


def _save_manifest(manifest: dict) -> None:
    MODEL_WEIGHTS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MODEL_WEIGHTS_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _fetch_one(model_id: str, force: bool, manifest: dict, verify_load: bool) -> dict:
    """Fetch (or confirm cached) one model. Returns its manifest entry."""
    s = models.spec(model_id)
    dest_dir = model_weights_dir(model_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    entry: dict = {
        "model_id": s.model_id,
        "arch": s.arch,
        "source": s.source,
        "image_size": s.image_size,
        "embed_dim": s.embed_dim,
    }

    if s.source == "huggingface":
        # download the single .pt file (huggingface_hub handles resume + caching)
        t0 = time.time()
        print(f"[{model_id}] downloading {s.hf_repo}/{s.hf_filename} ...")
        local_path = Path(
            hf_hub_download(
                repo_id=s.hf_repo,
                filename=s.hf_filename,
                local_dir=str(dest_dir),
                force_download=force,
            )
        )
        dt = time.time() - t0
        sz = local_path.stat().st_size
        print(f"[{model_id}] downloaded {sz / 1e6:.1f} MB in {dt:.1f}s -> {local_path}")
        sha = _sha256(local_path)
        entry.update(
            {
                "hf_repo": s.hf_repo,
                "hf_filename": s.hf_filename,
                "weights_path": str(local_path),
                "file_size_bytes": sz,
                "sha256": sha,
            }
        )
    elif s.source == "open_clip_pretrained":
        # open_clip handles its own cache via `pretrained=<tag>`. Trigger the
        # download by creating the model on CPU once; record the cache path
        # if we can extract it. open_clip 3.x stores under HF cache, exact
        # path varies; we just record the tag for reproducibility.
        import open_clip

        t0 = time.time()
        print(f"[{model_id}] open_clip create_model(arch={s.arch}, pretrained={s.pretrained}) ...")
        _model, _, _ = open_clip.create_model_and_transforms(
            s.arch, pretrained=s.pretrained, device="cpu"
        )
        del _model
        dt = time.time() - t0
        print(f"[{model_id}] ready (cache hit or fresh download) in {dt:.1f}s")
        entry.update(
            {
                "open_clip_pretrained_tag": s.pretrained,
                "weights_path": None,
                "file_size_bytes": None,
                "sha256": None,
            }
        )
    else:
        raise ValueError(f"unknown source {s.source!r}")

    entry["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # verify the model actually loads + forward-passes a zero tensor on MPS
    if verify_load:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[{model_id}] verifying load on {device} ...")
        t0 = time.time()
        model, _preprocess, tokenizer = models.load(
            model_id, weights_path=entry.get("weights_path"), device=device
        )
        # forward pass: encode a zero image and a one-token text, confirm
        # shapes match the spec's embed_dim
        with torch.no_grad():
            zero_img = torch.zeros(1, 3, s.image_size, s.image_size, device=device)
            img_emb = model.encode_image(zero_img)
            text_tokens = tokenizer(["test"]).to(device)
            txt_emb = model.encode_text(text_tokens)
        dt = time.time() - t0
        assert img_emb.shape[-1] == s.embed_dim, (
            f"{model_id}: image embed dim {img_emb.shape[-1]} != spec {s.embed_dim}"
        )
        assert txt_emb.shape[-1] == s.embed_dim, (
            f"{model_id}: text embed dim {txt_emb.shape[-1]} != spec {s.embed_dim}"
        )
        print(
            f"[{model_id}] verified in {dt:.1f}s: "
            f"image_emb {tuple(img_emb.shape)}, text_emb {tuple(txt_emb.shape)}"
        )
        entry["verified_load_device"] = device
        del model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch model weights + write manifest.")
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order. "
             "Used to pick the default model_id when --models is omitted.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model_ids to fetch; default = YAML model_id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if SHA256 matches manifest.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-download load + forward-pass on MPS.",
    )
    args = parser.parse_args()

    if args.models:
        targets = args.models
    else:
        cfg = config.load_experiment(args.experiment)
        targets = [config.model_id_of(cfg)]
    manifest = _load_manifest()

    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for mid in targets:
        prev = manifest["models"].get(mid)
        # idempotent: if not --force and we have an HF file with matching SHA
        # already on disk, skip download but still verify load
        if not args.force and prev and prev.get("source") == "huggingface":
            wp = prev.get("weights_path")
            if wp and Path(wp).exists() and _sha256(Path(wp)) == prev.get("sha256"):
                print(f"[{mid}] cached + SHA matches; skipping download.")
                # still re-verify load (cheap, catches model-loader regressions)
                if not args.no_verify:
                    entry = _fetch_one(
                        mid, force=False, manifest=manifest, verify_load=True
                    )
                    # preserve old fetched_at since we didn't refetch
                    entry["fetched_at"] = prev.get("fetched_at", entry["fetched_at"])
                    manifest["models"][mid] = entry
                continue
        entry = _fetch_one(
            mid, force=args.force, manifest=manifest, verify_load=not args.no_verify
        )
        manifest["models"][mid] = entry
        _save_manifest(manifest)
        print(f"[{mid}] manifest updated.\n")

    _save_manifest(manifest)
    print(f"wrote {MODEL_WEIGHTS_MANIFEST}")


if __name__ == "__main__":
    main()
