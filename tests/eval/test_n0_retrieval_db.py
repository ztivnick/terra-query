"""DB-backed retrieval evaluation: same fixture as the in-memory path,
asserts the metrics match."""

from __future__ import annotations

import json

import numpy as np
import pytest

from terra_query.eval import n0_retrieval
from terra_query.vector_store import loader


def _seed_two_cycle_fixture(db_conn, monkeypatch, tmp_path,
                            experiment_id="test-exp",
                            model_id="test-model", bands="rgb"):
    """10 chips x 2 cycles, vectors engineered so chip 7 dominates the query."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed=13)
    embed_dim = 768
    n_chips = 10

    base = rng.standard_normal((n_chips, embed_dim), dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # use chip 7's vector as the query later
    query_vec = base[7].copy()

    cycles = ["2018", "2022"]
    chip_index = {"cycles": []}
    for cycle in cycles:
        chips = []
        chip_ids = []
        for i in range(n_chips):
            row, col = divmod(i, 4)
            cid = f"naip_{cycle}_r{row:03d}_c{col:03d}"
            chip_ids.append(cid)
            x0 = 333424.0 + col * 224.0
            y0 = 5139120.0 + row * 224.0
            chips.append({
                "chip_id": cid,
                "row": row,
                "col": col,
                "bbox_26916": [x0, y0, x0 + 224.0, y0 + 224.0],
                "center_26916": [x0 + 112.0, y0 + 112.0],
                "center_wgs84": [0.0, 0.0],
                "inside_aoi": True,
            })
        chip_index["cycles"].append({
            "year": cycle,
            "source_cog": "synthetic",
            "native_pixel_size_m": 1.0,
            "native_chip_array_shape": [224, 224, 4],
            "chips": chips,
        })
        npy_path = embeddings_dir / f"{model_id}__{bands}__{cycle}.npy"
        json_path = embeddings_dir / f"{model_id}__{bands}__{cycle}.json"
        np.save(npy_path, base)
        json_path.write_text(json.dumps({
            "model_id": model_id, "arch": "synthetic", "bands": bands,
            "cycle": cycle, "embed_dim": embed_dim, "image_size": 224,
            "n_chips": n_chips, "chip_ids": chip_ids,
        }))

    # the loader resolves embedding paths via the experiment-scoped resolver;
    # patch both spellings to point at the test's tmp dir.
    monkeypatch.setattr(loader, "embeddings_npy",
        lambda exp, m, b, c: embeddings_dir / f"{m}__{b}__{c}.npy")
    monkeypatch.setattr(loader, "embeddings_json",
        lambda exp, m, b, c: embeddings_dir / f"{m}__{b}__{c}.json")
    for cycle in cycles:
        loader.load_cycle(db_conn, experiment_id, model_id, bands, cycle,
                          chip_index=chip_index)
    return {
        "experiment_id": experiment_id,
        "model_id": model_id, "bands": bands, "cycles": cycles,
        "chip_index": chip_index, "query_vec": query_vec,
    }


def test_chip_location_to_index_map_round_trip():
    """The map is the inverse of `r{row:03d}_c{col:03d}` over the location list."""
    chip_index = {
        "cycles": [{
            "year": "2099",
            "chips": [
                {"chip_id": f"naip_2099_r{r:03d}_c{c:03d}", "row": r, "col": c,
                 "bbox_26916": [0, 0, 1, 1], "center_26916": [0.5, 0.5],
                 "center_wgs84": [0, 0], "inside_aoi": True}
                for r in range(3) for c in range(4)
            ],
        }],
    }
    m = n0_retrieval.chip_location_to_index_map(chip_index)
    assert len(m) == 12
    assert m["r000_c000"] == 0
    assert m["r000_c003"] == 3
    assert m["r002_c003"] == 11


def test_evaluate_concept_via_db_matches_in_memory(db_conn, monkeypatch, tmp_path):
    """DB-backed and numpy paths produce the same metrics on a known fixture."""
    fx = _seed_two_cycle_fixture(db_conn, monkeypatch, tmp_path)
    chip_loc_to_idx = n0_retrieval.chip_location_to_index_map(fx["chip_index"])
    gt = {chip_loc_to_idx["r001_c003"]}  # chip 7 in row/col

    # in-memory path
    embeddings_by_cycle = {
        c: np.load(tmp_path / "embeddings" / f"{fx['model_id']}__{fx['bands']}__{c}.npy")
        for c in fx["cycles"]
    }
    mem_res = n0_retrieval.evaluate_concept(
        "waterfall", fx["query_vec"], embeddings_by_cycle,
        gt_locations=gt, top_k_max=10,
    )

    # DB-backed path
    db_res = n0_retrieval.evaluate_concept_via_db(
        "waterfall", fx["query_vec"], chip_loc_to_idx,
        gt_locations=gt, top_k_max=10,
        model_id=fx["model_id"], bands=fx["bands"], conn=db_conn,
        overfetch=30,
    )

    assert db_res.top_k_indices[0] == mem_res.top_k_indices[0]
    assert db_res.top_k_indices[0] == chip_loc_to_idx["r001_c003"]
    assert db_res.hit_at_k == mem_res.hit_at_k
    assert db_res.recall_at_k == mem_res.recall_at_k
    assert db_res.mrr == pytest.approx(mem_res.mrr)
    # top_k_indices should agree for at least the top-1 (HNSW is approximate);
    # at 10 rows it should be effectively exact
    assert db_res.top_k_indices[:5] == mem_res.top_k_indices[:5]


def test_evaluate_concept_via_db_empty_gt(db_conn, monkeypatch, tmp_path):
    """Empty GT -> zero metrics but top_k_indices still populated."""
    fx = _seed_two_cycle_fixture(db_conn, monkeypatch, tmp_path)
    chip_loc_to_idx = n0_retrieval.chip_location_to_index_map(fx["chip_index"])

    res = n0_retrieval.evaluate_concept_via_db(
        "abandoned_cabin", fx["query_vec"], chip_loc_to_idx,
        gt_locations=set(), top_k_max=5,
        model_id=fx["model_id"], bands=fx["bands"], conn=db_conn,
        overfetch=30,
    )
    assert res.n_gt == 0
    assert res.mrr == 0.0
    assert all(v == 0.0 for v in res.hit_at_k.values())
    assert all(v == 0.0 for v in res.recall_at_k.values())
    assert len(res.top_k_indices) == 5
