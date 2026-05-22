"""Bulk load per-cycle embedding .npy files into chip_embeddings.

Idempotent via ON CONFLICT DO UPDATE on the primary key
(model_id, bands, chip_id). One row per (model_id, bands, chip_id);
chip_location_id is the year-stripped grid id (e.g. "r034_c028") so
the search SQL can max-pool across cycles at query time.

Reads:
- chip_index.json for footprint / center / inside_aoi per chip
- embeddings/<model>__<bands>__<cycle>.npy for the vectors
- embeddings/<model>__<bands>__<cycle>.json for chip_id order
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psycopg

from terra_query.core.paths import (
    chip_index_json,
    embeddings_json,
    embeddings_npy,
)

_INSERT_SQL = """
INSERT INTO chip_embeddings
    (model_id, bands, chip_id, chip_location_id, source_cycle, inside_aoi,
     embedding, footprint, center_26916)
VALUES
    (%s, %s, %s, %s, %s, %s, %s,
     ST_GeomFromText(%s, 26916), ST_GeomFromText(%s, 26916))
ON CONFLICT (model_id, bands, chip_id) DO UPDATE SET
    chip_location_id = EXCLUDED.chip_location_id,
    source_cycle     = EXCLUDED.source_cycle,
    inside_aoi       = EXCLUDED.inside_aoi,
    embedding        = EXCLUDED.embedding,
    footprint        = EXCLUDED.footprint,
    center_26916     = EXCLUDED.center_26916
"""


def _polygon_wkt_from_bbox(bbox: list[float]) -> str:
    w, s, e, n = bbox
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _point_wkt(xy: list[float]) -> str:
    x, y = xy
    return f"POINT({x} {y})"


def _chip_location_id(chip_id: str, cycle: str) -> str:
    """Strip the source/year prefix; matches the n0_retrieval convention."""
    prefix = f"naip_{cycle}_"
    if not chip_id.startswith(prefix):
        raise ValueError(f"chip_id {chip_id!r} does not start with {prefix!r}")
    return chip_id[len(prefix) :]


def _load_chip_index(experiment_id: str, path: Path | None = None) -> dict:
    """Read the chip_index.json for an experiment.

    `path` overrides the resolver (used by tests with synthetic fixtures).
    """
    p = path if path is not None else chip_index_json(experiment_id)
    return json.loads(p.read_text())


def _cycle_block(chip_index: dict, cycle: str) -> dict:
    for c in chip_index["cycles"]:
        if c["year"] == cycle:
            return c
    raise KeyError(f"cycle {cycle!r} not in chip index")


def load_cycle(
    conn: psycopg.Connection,
    experiment_id: str,
    model_id: str,
    bands: str,
    cycle: str,
    chip_index: dict | None = None,
    batch_size: int = 1000,
) -> int:
    """Load one (experiment, model_id, bands, cycle) cell into the DB. Returns row count."""
    npy_path = embeddings_npy(experiment_id, model_id, bands, cycle)
    json_path = embeddings_json(experiment_id, model_id, bands, cycle)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"missing embeddings for ({model_id}, {bands}, {cycle})")

    embeddings = np.load(npy_path)
    sidecar = json.loads(json_path.read_text())
    chip_ids: list[str] = sidecar["chip_ids"]
    if embeddings.shape[0] != len(chip_ids):
        raise RuntimeError(
            f"embedding rows {embeddings.shape[0]} != sidecar chip_ids {len(chip_ids)} "
            f"for ({model_id}, {bands}, {cycle})"
        )
    if embeddings.dtype != np.float32:
        # the sidecar protocol guarantees float32; refuse anything else
        raise RuntimeError(f"expected float32 embeddings, got {embeddings.dtype}")

    if chip_index is None:
        chip_index = _load_chip_index(experiment_id)
    cycle_chips = _cycle_block(chip_index, cycle)["chips"]
    chip_by_id = {c["chip_id"]: c for c in cycle_chips}

    rows: list[tuple] = []
    for i, cid in enumerate(chip_ids):
        c = chip_by_id.get(cid)
        if c is None:
            raise RuntimeError(f"chip_id {cid!r} in sidecar but not in chip_index cycle {cycle}")
        rows.append(
            (
                model_id,
                bands,
                cid,
                _chip_location_id(cid, cycle),
                cycle,
                bool(c["inside_aoi"]),
                embeddings[i],
                _polygon_wkt_from_bbox(c["bbox_26916"]),
                _point_wkt(c["center_26916"]),
            )
        )

    with conn.cursor() as cur:
        # executemany rather than COPY: keeps ON CONFLICT idempotency.
        # batched for memory bounding only; 13k rows is small enough that
        # one call would also work.
        for start in range(0, len(rows), batch_size):
            cur.executemany(_INSERT_SQL, rows[start : start + batch_size])
    conn.commit()
    return len(rows)


def load_all(
    conn: psycopg.Connection,
    experiment_id: str,
    model_id: str,
    bands: str,
    cycles: list[str] | None = None,
) -> dict[str, int]:
    """Load every requested cycle; defaults to all cycles in the chip index.

    Returns a dict cycle -> row count.
    """
    chip_index = _load_chip_index(experiment_id)
    if cycles is None:
        cycles = [c["year"] for c in chip_index["cycles"]]

    results: dict[str, int] = {}
    for cycle in cycles:
        t0 = time.time()
        n = load_cycle(conn, experiment_id, model_id, bands, cycle, chip_index=chip_index)
        dt = time.time() - t0
        print(f"[load {model_id}/{bands}/{cycle}] {n} rows in {dt:.2f}s")
        results[cycle] = n
    return results
