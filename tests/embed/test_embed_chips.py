"""embed_chips unit tests. Exercises the single-chip path + the full-cycle
artifact path against the production model."""

from __future__ import annotations

import json

import numpy as np
import pytest

from terra_query.core.paths import (
    CHIP_INDEX_JSON,
    MODEL_WEIGHTS_MANIFEST,
    naip_cog,
)
from terra_query.embed import models
from terra_query.embed.embed_chips import (
    BANDS_TABLE,
    _chip_box_from_record,
    _get_cycle_block,
    embed_one_chip,
)


@pytest.fixture(scope="module")
def chip_index() -> dict:
    if not CHIP_INDEX_JSON.exists():
        pytest.skip("chip index missing; run `build_chip_index` first")
    return json.loads(CHIP_INDEX_JSON.read_text())


@pytest.fixture(scope="module")
def production_model_loaded():
    if not MODEL_WEIGHTS_MANIFEST.exists():
        pytest.skip("weights manifest not present; run fetch_weights first")
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess, tokenizer = models.load_production(device=device)
    embed_dim = models.spec(models.PRODUCTION_MODEL_ID).embed_dim
    return model, preprocess, tokenizer, device, embed_dim


def test_bands_table_covers_rgb_and_cir():
    assert "rgb" in BANDS_TABLE and "cir" in BANDS_TABLE
    assert BANDS_TABLE["rgb"] == (1, 2, 3)
    assert BANDS_TABLE["cir"] == (4, 1, 2)


def test_chip_box_from_record_round_trip(chip_index):
    ch = chip_index["cycles"][0]["chips"][100]
    box = _chip_box_from_record(ch)
    assert (box.west, box.south) == (ch["bbox_26916"][0], ch["bbox_26916"][1])
    assert box.east == ch["bbox_26916"][2]
    assert box.north == ch["bbox_26916"][3]
    assert (box.row, box.col) == (ch["row"], ch["col"])


def test_get_cycle_block_known_year(chip_index):
    block = _get_cycle_block(chip_index, "2022")
    assert block["year"] == "2022"
    assert len(block["chips"]) == chip_index["n_rows"] * chip_index["n_cols"]


def test_get_cycle_block_missing_year_raises(chip_index):
    with pytest.raises(KeyError):
        _get_cycle_block(chip_index, "1999")


def test_embed_one_chip_bond_falls_rgb(chip_index, production_model_loaded):
    """Embed Bond Falls' 2022 RGB chip with the production model -> unit vector
    of the expected dimensionality."""
    model, preprocess, _tokenizer, device, embed_dim = production_model_loaded
    bf_entry = next(e for e in chip_index["eval_lookup"]["bond-falls"] if e["year"] == "2022")
    chips_by_id = {c["chip_id"]: c for c in _get_cycle_block(chip_index, "2022")["chips"]}
    bf_chip = chips_by_id[bf_entry["chip_id"]]
    emb = embed_one_chip(bf_chip, naip_cog("2022"), "rgb", model, preprocess, device)
    assert emb.shape == (embed_dim,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_embed_one_chip_bond_falls_cir(chip_index, production_model_loaded):
    """Same chip, CIR bands. PIL packs (NIR, R, G) as a 3-channel image and
    the preprocessor does its standard normalization. We don't expect CIR
    semantics to match the CLIP training distribution, but the output must
    still be a unit-normalized vector of the right shape."""
    model, preprocess, _tokenizer, device, embed_dim = production_model_loaded
    bf_entry = next(e for e in chip_index["eval_lookup"]["bond-falls"] if e["year"] == "2022")
    chips_by_id = {c["chip_id"]: c for c in _get_cycle_block(chip_index, "2022")["chips"]}
    bf_chip = chips_by_id[bf_entry["chip_id"]]
    emb = embed_one_chip(bf_chip, naip_cog("2022"), "cir", model, preprocess, device)
    assert emb.shape == (embed_dim,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


# === full-cycle embed artifact (production model only) ===


def _load_npy_and_sidecar(model_id: str, bands: str, cycle: str):
    from terra_query.core.paths import embeddings_json, embeddings_npy

    npy = embeddings_npy(model_id, bands, cycle)
    js = embeddings_json(model_id, bands, cycle)
    if not (npy.exists() and js.exists()):
        pytest.skip(
            f"embedding artifact missing for ({model_id}, {bands}, {cycle}); "
            f"run embed_chips first"
        )
    return np.load(npy), json.loads(js.read_text())


def test_production_rgb_2022_artifact_shape_and_norms(chip_index):
    """Production embed cell shape + L2-unit-norm contract."""
    embed_dim = models.spec(models.PRODUCTION_MODEL_ID).embed_dim
    arr, sc = _load_npy_and_sidecar(models.PRODUCTION_MODEL_ID, "rgb", "2022")
    cycle_2022 = _get_cycle_block(chip_index, "2022")
    assert arr.shape == (len(cycle_2022["chips"]), embed_dim)
    assert arr.dtype == np.float32
    assert not np.isnan(arr).any()
    norms = np.linalg.norm(arr, axis=1)
    assert (np.abs(norms - 1.0) < 1e-3).all()
    assert sc["n_chips"] == arr.shape[0]
    assert sc["embed_dim"] == arr.shape[1]
    assert sc["model_id"] == models.PRODUCTION_MODEL_ID
    assert sc["bands"] == "rgb"
    assert sc["cycle"] == "2022"


def test_production_rgb_2022_sidecar_chip_ids_match_index_order(chip_index):
    """Sidecar chip_ids must match cycles[year=2022].chips order exactly so
    row i of the .npy maps to chip_ids[i] in the chip index."""
    _arr, sc = _load_npy_and_sidecar(models.PRODUCTION_MODEL_ID, "rgb", "2022")
    cycle_2022 = _get_cycle_block(chip_index, "2022")
    expected_ids = [c["chip_id"] for c in cycle_2022["chips"]]
    assert sc["chip_ids"] == expected_ids


def test_production_rgb_2022_pins_chip_index_and_weights_provenance(chip_index):
    """Sidecar pins which chip_index and which weights produced this embedding.
    Idempotency uses the EMBEDDING-RELEVANT subset (excludes generated_at +
    eval_lookup) so adding new eval features doesn't false-flag existing
    embeddings as stale. See terra_query.embed.embed_chips._chip_index_checksum."""
    from terra_query.core.paths import MODEL_WEIGHTS_MANIFEST
    from terra_query.embed.embed_chips import _chip_index_checksum

    _arr, sc = _load_npy_and_sidecar(models.PRODUCTION_MODEL_ID, "rgb", "2022")
    assert sc["chip_index_sha256"] == _chip_index_checksum()
    wmf = json.loads(MODEL_WEIGHTS_MANIFEST.read_text())
    assert sc["weights_sha256"] == wmf["models"][models.PRODUCTION_MODEL_ID]["sha256"]


# === parallelization equivalence ===


def test_dataloader_path_matches_serial_for_subset(chip_index, production_model_loaded):
    """DataLoader-backed embedding must match the serial path within fp32
    tolerance for a small subset of chips. Tests both num_workers=0
    (in-process) and num_workers>0 (multi-process)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from terra_query.embed.embed_chips import (
        BANDS_TABLE,
        _ChipDataset,
        embed_one_chip,
    )

    model, preprocess, _tokenizer, device, _embed_dim = production_model_loaded
    chips = _get_cycle_block(chip_index, "2022")["chips"][:8]
    cog = naip_cog("2022")

    serial = np.stack(
        [embed_one_chip(ch, cog, "rgb", model, preprocess, device) for ch in chips],
        axis=0,
    )

    def run_parallel(num_workers: int) -> np.ndarray:
        dataset = _ChipDataset(chips, cog, BANDS_TABLE["rgb"], preprocess)
        loader = DataLoader(
            dataset, batch_size=4, num_workers=num_workers, shuffle=False
        )
        out = []
        for tb, _ids in loader:
            tb = tb.to(device)
            with torch.no_grad():
                e = model.encode_image(tb)
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().float().numpy())
        return np.concatenate(out, axis=0)

    in_proc = run_parallel(num_workers=0)
    assert in_proc.shape == serial.shape
    np.testing.assert_allclose(serial, in_proc, atol=1e-4)
    multi = run_parallel(num_workers=2)
    np.testing.assert_allclose(serial, multi, atol=1e-4)
