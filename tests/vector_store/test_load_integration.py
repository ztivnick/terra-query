"""Integration tests for the loader. Skips if the DB is unreachable."""

from __future__ import annotations

import json

import numpy as np
import pytest

from terra_query.vector_store import loader


def _make_fixture(tmp_path, experiment_id="test-exp", model_id="test-model",
                  bands="rgb", cycle="2099", n_chips=10):
    """Write a synthetic chip_index + embeddings .npy + sidecar under tmp_path."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    chips = []
    embed_dim = 768
    rng = np.random.default_rng(seed=42)
    vectors = rng.standard_normal((n_chips, embed_dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
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
            "inside_aoi": (i % 2 == 0),
        })

    chip_index = {
        "cycles": [
            {
                "year": cycle,
                "source_cog": "synthetic",
                "native_pixel_size_m": 1.0,
                "native_chip_array_shape": [224, 224, 4],
                "chips": chips,
            }
        ]
    }

    npy_path = embeddings_dir / f"{model_id}__{bands}__{cycle}.npy"
    json_path = embeddings_dir / f"{model_id}__{bands}__{cycle}.json"
    np.save(npy_path, vectors)
    json_path.write_text(json.dumps({
        "model_id": model_id,
        "arch": "synthetic",
        "bands": bands,
        "cycle": cycle,
        "embed_dim": embed_dim,
        "image_size": 224,
        "n_chips": n_chips,
        "chip_ids": chip_ids,
    }))
    return {
        "chip_index": chip_index,
        "vectors": vectors,
        "chip_ids": chip_ids,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "bands": bands,
        "cycle": cycle,
        "embeddings_dir": embeddings_dir,
    }


def test_load_cycle_round_trip(db_conn, tmp_path, monkeypatch):
    """10 synthetic chips load + round-trip with bit-perfect vectors."""
    fx = _make_fixture(tmp_path)

    # redirect the loader's path lookups at our synthetic dir
    monkeypatch.setattr(loader, "embeddings_npy",
        lambda exp, m, b, c: fx["embeddings_dir"] / f"{m}__{b}__{c}.npy")
    monkeypatch.setattr(loader, "embeddings_json",
        lambda exp, m, b, c: fx["embeddings_dir"] / f"{m}__{b}__{c}.json")

    n = loader.load_cycle(
        db_conn,
        experiment_id=fx["experiment_id"],
        model_id=fx["model_id"],
        bands=fx["bands"],
        cycle=fx["cycle"],
        chip_index=fx["chip_index"],
    )
    assert n == 10

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chip_embeddings")
        assert cur.fetchone()[0] == 10

        # every chip row's embedding matches its source vector exactly
        for i, cid in enumerate(fx["chip_ids"]):
            cur.execute(
                "SELECT embedding, chip_location_id, source_cycle, inside_aoi, "
                "ST_AsText(footprint), ST_AsText(center_26916) "
                "FROM chip_embeddings WHERE chip_id = %s",
                (cid,),
            )
            emb, loc, src_cycle, inside, fp_wkt, ct_wkt = cur.fetchone()
            assert np.array_equal(np.asarray(emb, dtype=np.float32), fx["vectors"][i])
            assert loc == cid.replace(f"naip_{fx['cycle']}_", "")
            assert src_cycle == fx["cycle"]
            assert inside == (i % 2 == 0)
            assert "POLYGON" in fp_wkt
            assert "POINT" in ct_wkt


def test_load_cycle_is_idempotent(db_conn, tmp_path, monkeypatch):
    """Re-running load_cycle upserts on the primary key."""
    fx = _make_fixture(tmp_path)
    monkeypatch.setattr(loader, "embeddings_npy",
        lambda exp, m, b, c: fx["embeddings_dir"] / f"{m}__{b}__{c}.npy")
    monkeypatch.setattr(loader, "embeddings_json",
        lambda exp, m, b, c: fx["embeddings_dir"] / f"{m}__{b}__{c}.json")

    loader.load_cycle(db_conn, fx["experiment_id"], fx["model_id"], fx["bands"],
                      fx["cycle"], chip_index=fx["chip_index"])
    loader.load_cycle(db_conn, fx["experiment_id"], fx["model_id"], fx["bands"],
                      fx["cycle"], chip_index=fx["chip_index"])
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chip_embeddings")
        assert cur.fetchone()[0] == 10


def test_load_cycle_rejects_unknown_chip_id(db_conn, tmp_path, monkeypatch):
    """A sidecar listing chips not in the chip_index is a hard error."""
    fx = _make_fixture(tmp_path)
    # corrupt the sidecar with an extra chip id
    sidecar_path = fx["embeddings_dir"] / f"{fx['model_id']}__{fx['bands']}__{fx['cycle']}.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["chip_ids"][0] = "naip_2099_r999_c999"  # not in chip_index
    sidecar_path.write_text(json.dumps(sidecar))
    monkeypatch.setattr(loader, "embeddings_npy",
        lambda exp, m, b, c: fx["embeddings_dir"] / f"{m}__{b}__{c}.npy")
    monkeypatch.setattr(loader, "embeddings_json",
        lambda exp, m, b, c: fx["embeddings_dir"] / f"{m}__{b}__{c}.json")
    with pytest.raises(RuntimeError, match="not in chip_index"):
        loader.load_cycle(db_conn, fx["experiment_id"], fx["model_id"], fx["bands"],
                          fx["cycle"], chip_index=fx["chip_index"])
