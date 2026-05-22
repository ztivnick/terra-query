"""Integration tests for the search API: ANN, max-pool, spatial filters."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from terra_query.vector_store import loader, search


def _seed_fixture(db_conn, monkeypatch, tmp_path,
                  experiment_id="test-exp",
                  model_id="test-model", bands="rgb",
                  n_chips=10, n_cycles=3):
    """Load `n_cycles` cycles of `n_chips` chips with known geometry + embeddings.

    Embeddings: each chip-location gets the SAME unit vector across cycles
    except that in cycle "2099" the vector for chip 5 is perturbed to make
    that cycle "win" the max-pool for that chip.
    """
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed=0)

    # one shared base vector per chip-location
    embed_dim = 768
    base = rng.standard_normal((n_chips, embed_dim), dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    cycles = [f"209{i}" for i in range(n_cycles)]
    chip_index = {"cycles": []}
    for cycle_idx, cycle in enumerate(cycles):
        chips = []
        chip_ids = []
        vectors = base.copy()
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
                "inside_aoi": (i < n_chips // 2),
            })
        # perturb chip 5 in the LAST cycle so it has the highest score for
        # a query vector close to chip 5's base
        if cycle_idx == n_cycles - 1:
            vectors[5] = base[5] * 0.95 + base[0] * 0.05
            vectors[5] /= np.linalg.norm(vectors[5])
        chip_index["cycles"].append({
            "year": cycle,
            "source_cog": "synthetic",
            "native_pixel_size_m": 1.0,
            "native_chip_array_shape": [224, 224, 4],
            "chips": chips,
        })
        npy_path = embeddings_dir / f"{model_id}__{bands}__{cycle}.npy"
        json_path = embeddings_dir / f"{model_id}__{bands}__{cycle}.json"
        np.save(npy_path, vectors)
        json_path.write_text(json.dumps({
            "model_id": model_id, "arch": "synthetic", "bands": bands,
            "cycle": cycle, "embed_dim": embed_dim, "image_size": 224,
            "n_chips": n_chips, "chip_ids": chip_ids,
        }))

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
        "base_vectors": base, "n_chips": n_chips, "chip_index": chip_index,
    }


def test_search_returns_top_k_ranked(db_conn, monkeypatch, tmp_path):
    fx = _seed_fixture(db_conn, monkeypatch, tmp_path)
    # query vector identical to chip 7's base -> chip 7 must come back rank 1
    q = fx["base_vectors"][7]
    hits = search.search(
        q, top_k=3, model_id=fx["model_id"], bands=fx["bands"],
        conn=db_conn, overfetch=30,
    )
    assert len(hits) == 3
    assert hits[0].chip_location_id == "r001_c003"  # chip 7 = row 1, col 3
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    # scores monotone non-increasing
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_max_pools_across_cycles(db_conn, monkeypatch, tmp_path):
    fx = _seed_fixture(db_conn, monkeypatch, tmp_path)
    # chip 5's perturbed vector (last cycle) is closer to chip 0 than chip 5's
    # base, so it dominates the max-pool for chip 5 when we query for chip 0.
    q = fx["base_vectors"][0]
    hits = search.search(
        q, top_k=10, model_id=fx["model_id"], bands=fx["bands"],
        conn=db_conn, overfetch=30,
    )
    # rank 1 is the chip whose vector EQUALS q -> r000_c000
    assert hits[0].chip_location_id == "r000_c000"
    # the perturbed chip 5 should have its winning_cycle = last cycle
    chip5_hits = [h for h in hits if h.chip_location_id == "r001_c001"]
    assert len(chip5_hits) == 1
    assert chip5_hits[0].winning_cycle == fx["cycles"][-1]


def test_bbox_filter_restricts_results(db_conn, monkeypatch, tmp_path):
    fx = _seed_fixture(db_conn, monkeypatch, tmp_path)
    q = fx["base_vectors"][7]  # would normally rank chip 7 first
    # bbox covers chips 0..3 only (row 0). chip 7 (row 1) is excluded.
    bbox = (333424.0, 5139120.0, 333424.0 + 224.0 * 4 + 1, 5139120.0 + 223.0)
    hits = search.search(
        q, top_k=10, bbox_26916=bbox,
        model_id=fx["model_id"], bands=fx["bands"],
        conn=db_conn, overfetch=30,
    )
    assert hits, "bbox-filtered query returned nothing"
    locs = {h.chip_location_id for h in hits}
    # only row-0 locations
    assert locs == {"r000_c000", "r000_c001", "r000_c002", "r000_c003"}


def test_near_filter_uses_dwithin(db_conn, monkeypatch, tmp_path):
    fx = _seed_fixture(db_conn, monkeypatch, tmp_path)
    q = fx["base_vectors"][7]
    # near chip 7's center, radius covers only chip 7 (~224m grid; 50m radius)
    chip7 = fx["chip_index"]["cycles"][0]["chips"][7]
    nx, ny = chip7["center_26916"]
    hits = search.search(
        q, top_k=10, near_26916=(nx, ny, 50.0),
        model_id=fx["model_id"], bands=fx["bands"],
        conn=db_conn, overfetch=30,
    )
    assert {h.chip_location_id for h in hits} == {"r001_c003"}


def test_inside_aoi_only_filters_out(db_conn, monkeypatch, tmp_path):
    fx = _seed_fixture(db_conn, monkeypatch, tmp_path)
    q = fx["base_vectors"][7]  # chip 7 has inside_aoi=False in the fixture
    hits = search.search(
        q, top_k=10, inside_aoi_only=True,
        model_id=fx["model_id"], bands=fx["bands"],
        conn=db_conn, overfetch=30,
    )
    # chips 0..4 are inside_aoi=True, chips 5..9 are False
    locs = {h.chip_location_id for h in hits}
    assert "r001_c003" not in locs  # chip 7
    for loc in locs:
        # r000_c000..r001_c000 i.e. chips 0..4
        assert int(loc[1:4]) * 4 + int(loc[6:9]) < 5


def test_search_uses_production_data_when_available(monkeypatch):
    """Smoke test against the production schema if it is populated.

    Skipped if chip_embeddings in `public` has no rows.
    """
    import psycopg

    from terra_query.embed.models import PRODUCTION_MODEL_ID
    from terra_query.vector_store.db import DEFAULT_DSN, DSN_ENV_VAR, connect

    import os
    dsn = os.environ.get(DSN_ENV_VAR, DEFAULT_DSN)
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM public.chip_embeddings "
                    "WHERE model_id = %s",
                    (PRODUCTION_MODEL_ID,),
                )
                n = cur.fetchone()[0]
    except Exception as e:
        pytest.skip(f"DB unreachable: {e}")
    if n == 0:
        pytest.skip("production chip_embeddings empty; load via the CLI first")

    rng = np.random.default_rng(seed=0)
    q = rng.standard_normal(768, dtype=np.float32)
    q /= np.linalg.norm(q)
    t0 = time.time()
    hits = search.search(q, top_k=20)
    dt_ms = (time.time() - t0) * 1000
    assert len(hits) == 20
    assert dt_ms < 500, f"top-20 query latency {dt_ms:.0f} ms > 500 ms"
