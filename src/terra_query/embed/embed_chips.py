"""Embed every NAIP chip for one (model, bands, cycle) -> .npy + sidecar.

Reads chips lazily via `ingest.chips.read_chip` against the source COG at
the cycle's native resolution; the model preprocessor handles the resize
to model input. Output: a row-per-chip float32 .npy + a sidecar .json
describing the embedding artifact.

Embedding uses a PyTorch DataLoader with multi-worker chip prefetching so
the GPU stays fed while CPU does chip reads + preprocess in parallel.
Single-process MPS embedding (num_workers=0) is the deprecated baseline.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import Dataset

from terra_query.core.paths import (
    CHIP_INDEX_JSON,
    MODEL_WEIGHTS_MANIFEST,
    embeddings_json,
    embeddings_npy,
    naip_cog,
)
from terra_query.embed import encoder, models
from terra_query.ingest.chips import CIR_BANDS, RGB_BANDS, ChipBox, read_chip

BANDS_TABLE = {
    "rgb": RGB_BANDS,
    "cir": CIR_BANDS,
}

# default workers for the DataLoader. 0 means run in main process (slow,
# deprecated baseline). M5 has 10 CPU cores so 4 keeps headroom for the
# main process + GPU driver + system.
DEFAULT_NUM_WORKERS = 4


def _chip_box_from_record(ch: dict) -> ChipBox:
    return ChipBox(
        row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1]
    )


def _peak_rss_mb() -> float:
    import resource

    # macOS reports ru_maxrss in BYTES (per BSD-ish convention); linux in KB.
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    if sys.platform == "darwin":
        return val / (1024 * 1024)
    return val / 1024  # linux: KB -> MB


def embed_one_chip(chip_record: dict, cog_path: Path, bands: str, model, preprocess, device):
    """Read one chip's pixels via `read_chip`, encode -> (embed_dim,) float32.

    Used by tests and one-off sanity checks. Production embedding uses
    `embed_cycle` / `embed_cycles_for_combo` which batch via DataLoader.
    """
    if bands not in BANDS_TABLE:
        raise ValueError(f"unknown bands {bands!r}; known: {list(BANDS_TABLE)}")
    chip = _chip_box_from_record(chip_record)
    arr = read_chip(chip, cog_path, bands=BANDS_TABLE[bands])  # (3, h, w) uint8
    return encoder.encode_image(model, preprocess, arr, device)


def _load_chip_index() -> dict:
    return json.loads(CHIP_INDEX_JSON.read_text())


def _get_cycle_block(chip_index: dict, year: str) -> dict:
    for c in chip_index["cycles"]:
        if c["year"] == year:
            return c
    raise KeyError(f"cycle {year!r} not in chip index (have: {[c['year'] for c in chip_index['cycles']]})")


def _chip_index_checksum() -> str:
    """SHA256 of the EMBEDDING-RELEVANT parts of the chip index.

    Excludes `generated_at` (timestamp, changes every regen) and
    `eval_lookup` (grows when the eval set adds features; doesn't
    affect chip pixel content or grid). Including those would force
    false-positive re-embed cascades every time eval features are
    added or the index is regenerated cosmetically.
    """
    import hashlib

    idx = json.loads(CHIP_INDEX_JSON.read_text())
    # build a canonical embedding-relevant subset
    relevant = {k: v for k, v in idx.items() if k not in ("generated_at", "eval_lookup")}
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_up_to_date(model_id: str, bands: str, cycle: str, weights_entry: dict, index_sha: str) -> dict | None:
    """Return the existing sidecar dict if this cell is up to date, else None."""
    out_npy = embeddings_npy(model_id, bands, cycle)
    out_json = embeddings_json(model_id, bands, cycle)
    if not (out_npy.exists() and out_json.exists()):
        return None
    prev = json.loads(out_json.read_text())
    if prev.get("chip_index_sha256") != index_sha:
        return None
    if prev.get("weights_sha256") != weights_entry.get("sha256"):
        return None
    if prev.get("pretrained_tag") != weights_entry.get("open_clip_pretrained_tag"):
        return None
    return prev


class _ChipDataset(Dataset):
    """Reads a chip from its COG and preprocesses to a model-input tensor.

    Module-level (not nested) so DataLoader workers can pickle / unpickle
    instances when `num_workers > 0`. Workers spawn (Mac default) or fork,
    each opening the COG independently — rasterio supports concurrent reads
    of the same COG via separate file handles.
    """

    def __init__(self, chips, cog_path, bands_tuple, preprocess):
        # store cog_path as str to keep state purely picklable (Path objects
        # pickle fine, but str avoids any platform quirks across workers)
        self.chips = chips
        self.cog_path = str(cog_path)
        self.bands_tuple = bands_tuple
        self.preprocess = preprocess

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        ch = self.chips[idx]
        chip_box = _chip_box_from_record(ch)
        arr = read_chip(chip_box, Path(self.cog_path), bands=self.bands_tuple)
        img = encoder._to_pil_rgb(arr)
        tensor = self.preprocess(img)
        return tensor, ch["chip_id"]


def _embed_cycle_with_model(
    cycle: str,
    model_id: str,
    bands: str,
    model,
    preprocess,
    device: str,
    weights_entry: dict,
    index_sha: str,
    chip_index: dict,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Embed one cell with an already-loaded model. Writes .npy + sidecar.

    Uses a multi-worker DataLoader for chip reads + preprocess; the main
    process handles GPU forward passes only.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    cycle_block = _get_cycle_block(chip_index, cycle)
    cog = naip_cog(cycle)
    if not cog.exists():
        raise FileNotFoundError(f"NAIP COG missing for cycle {cycle}: {cog}")

    bands_tuple = BANDS_TABLE[bands]
    dataset = _ChipDataset(cycle_block["chips"], cog, bands_tuple, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # MPS does not benefit from pin_memory
        shuffle=False,  # preserve chip-id order for the sidecar
        drop_last=False,
    )

    chip_ids: list[str] = []
    embeddings_chunks: list[np.ndarray] = []
    n_total = len(cycle_block["chips"])
    log_every = max(1, batch_size * 16)

    t_embed = time.time()
    for tensor_batch, id_batch in loader:
        tensor_batch = tensor_batch.to(device)
        with torch.no_grad():
            emb = model.encode_image(tensor_batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings_chunks.append(emb.cpu().float().numpy())
        # id_batch from default_collate on strings is a list
        chip_ids.extend(list(id_batch))
        n_done = len(chip_ids)
        if n_done % log_every == 0 or n_done >= n_total:
            dt = time.time() - t_embed
            rate = n_done / dt if dt > 0 else 0.0
            print(
                f"[{model_id}/{bands}/{cycle}] embedded {n_done}/{n_total} "
                f"({rate:.1f} chips/s)"
            )

    dt_embed = time.time() - t_embed
    rate = len(chip_ids) / dt_embed if dt_embed > 0 else 0.0

    embeddings = np.concatenate(embeddings_chunks, axis=0).astype(np.float32, copy=False)
    assert embeddings.shape == (n_total, weights_entry["embed_dim"]), (
        f"unexpected shape {embeddings.shape}"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert (np.abs(norms - 1.0) < 1e-3).all(), (
        f"non-unit row norms: min={norms.min()}, max={norms.max()}"
    )

    out_npy = embeddings_npy(model_id, bands, cycle)
    out_json = embeddings_json(model_id, bands, cycle)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, embeddings)

    sidecar = {
        "model_id": model_id,
        "arch": weights_entry["arch"],
        "bands": bands,
        "cycle": cycle,
        "embed_dim": weights_entry["embed_dim"],
        "image_size": weights_entry["image_size"],
        "n_chips": int(embeddings.shape[0]),
        "chip_ids": chip_ids,
        "chip_index_sha256": index_sha,
        "weights_sha256": weights_entry.get("sha256"),
        "pretrained_tag": weights_entry.get("open_clip_pretrained_tag"),
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "embed_wall_clock_s": round(dt_embed, 2),
        "chips_per_sec": round(rate, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_json.write_text(json.dumps(sidecar, indent=2))

    print(
        f"[{model_id}/{bands}/{cycle}] wrote {out_npy} "
        f"({embeddings.shape[0]} x {embeddings.shape[1]}, "
        f"{dt_embed:.1f}s embed, {rate:.1f} chips/s, "
        f"peak_rss {sidecar['peak_rss_mb']:.0f} MB, "
        f"num_workers={num_workers})"
    )
    return sidecar


def embed_cycle(
    cycle: str,
    model_id: str,
    bands: str,
    batch_size: int = 16,
    force: bool = False,
    device: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict:
    """Embed all chips for one (model, bands, cycle) -> .npy + sidecar.

    Loads the model, embeds the one cycle, frees. For multi-cycle sweeps
    use `embed_cycles_for_combo` to share the model load across cycles.
    """
    import torch

    chip_index = _load_chip_index()
    index_sha = _chip_index_checksum()
    weights_manifest = json.loads(MODEL_WEIGHTS_MANIFEST.read_text())
    weights_entry = weights_manifest["models"][model_id]

    if not force:
        prev = _is_up_to_date(model_id, bands, cycle, weights_entry, index_sha)
        if prev is not None:
            print(f"[{model_id}/{bands}/{cycle}] up to date; skipping.")
            return prev

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"[{model_id}/{bands}/{cycle}] loading model on {device} "
        f"@ batch={batch_size} num_workers={num_workers}"
    )
    t_load = time.time()
    model, preprocess, _tokenizer = models.load(
        model_id, weights_path=weights_entry.get("weights_path"), device=device
    )
    print(f"[{model_id}/{bands}/{cycle}] model loaded in {time.time() - t_load:.1f}s")

    try:
        sidecar = _embed_cycle_with_model(
            cycle, model_id, bands, model, preprocess, device,
            weights_entry, index_sha, chip_index, batch_size, num_workers,
        )
    finally:
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return sidecar


def embed_cycles_for_combo(
    cycles: list[str],
    model_id: str,
    bands: str,
    batch_size: int = 16,
    force: bool = False,
    device: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> list[dict]:
    """Embed every (cycle) for one (model, bands) combo, sharing one model load.

    Skips any cell that's already up to date BEFORE loading the model, so a
    fully-cached combo costs no model load.
    """
    import torch

    chip_index = _load_chip_index()
    index_sha = _chip_index_checksum()
    weights_manifest = json.loads(MODEL_WEIGHTS_MANIFEST.read_text())
    weights_entry = weights_manifest["models"][model_id]

    # determine which cycles still need work
    to_do: list[str] = []
    sidecars: list[dict] = []
    for cycle in cycles:
        if force:
            to_do.append(cycle)
            continue
        prev = _is_up_to_date(model_id, bands, cycle, weights_entry, index_sha)
        if prev is not None:
            print(f"[{model_id}/{bands}/{cycle}] up to date; skipping.")
            sidecars.append(prev)
        else:
            to_do.append(cycle)

    if not to_do:
        return sidecars

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"[{model_id}/{bands}] loading model on {device} for {len(to_do)} cycle(s): "
        f"{to_do} (batch={batch_size}, num_workers={num_workers})"
    )
    t_load = time.time()
    model, preprocess, _tokenizer = models.load(
        model_id, weights_path=weights_entry.get("weights_path"), device=device
    )
    print(f"[{model_id}/{bands}] model loaded in {time.time() - t_load:.1f}s")

    try:
        for cycle in to_do:
            sc = _embed_cycle_with_model(
                cycle, model_id, bands, model, preprocess, device,
                weights_entry, index_sha, chip_index, batch_size, num_workers,
            )
            sidecars.append(sc)
    finally:
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return sidecars
