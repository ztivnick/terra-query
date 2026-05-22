"""Vector-store search: HNSW ANN, max-pool across cycles, optional spatial filter.

Query path: ANN -> overfetch rows -> GROUP BY chip_location_id with
MAX(score) -> top_k locations. Overfetch defaults to a multiple of
top_k; bump it via the kwarg if recall degrades.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import psycopg
from psycopg import sql

from terra_query.embed.models import PRODUCTION_MODEL_ID
from terra_query.vector_store.db import connect

# pgvector HNSW recall knob. The pgvector default (40) is too low for
# our scale: chips whose true cosine is highest can sit outside the
# search frontier and never enter the candidate set. Override via env.
EF_SEARCH_ENV = "TERRA_QUERY_HNSW_EF_SEARCH"
DEFAULT_EF_SEARCH = 1000


def _resolve_ef_search() -> int:
    raw = os.environ.get(EF_SEARCH_ENV)
    if not raw:
        return DEFAULT_EF_SEARCH
    return int(raw)


@dataclass(frozen=True)
class SearchHit:
    """One ranked result.

    `score` is the max cosine across cycles (cosine = 1 - distance).
    `winning_cycle` is the cycle that contributed the max for this place.
    `bbox_26916` is (xmin, ymin, xmax, ymax) extracted from the polygon.
    """

    chip_location_id: str
    score: float
    winning_cycle: str
    center_26916: tuple[float, float]
    bbox_26916: tuple[float, float, float, float]
    inside_aoi: bool


def _default_overfetch(top_k: int) -> int:
    # multiplier covers per-cycle expansion + safety; floor gives small top_k headroom
    return max(top_k * 12, 60)


def search(
    query_vec: np.ndarray,
    top_k: int = 10,
    bbox_26916: tuple[float, float, float, float] | None = None,
    near_26916: tuple[float, float, float] | None = None,
    inside_aoi_only: bool = False,
    model_id: str = PRODUCTION_MODEL_ID,
    bands: str = "rgb",
    overfetch: int | None = None,
    conn: psycopg.Connection | None = None,
) -> list[SearchHit]:
    """Top-K chip-locations by max-pooled cosine, with optional spatial filter.

    `query_vec` is a 1-D L2-normalized float array of length matching the
    stored embedding dim (768 for the production model).

    Spatial filters compose (AND): bbox AND/OR near AND/OR inside_aoi_only.

    If `conn` is None, a connection is opened from the env-var DSN.
    """
    if query_vec.ndim != 1:
        raise ValueError(f"query_vec must be 1-D, got shape {query_vec.shape}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    overfetch = overfetch if overfetch is not None else _default_overfetch(top_k)
    if overfetch < top_k:
        raise ValueError(f"overfetch {overfetch} must be >= top_k {top_k}")

    q = np.asarray(query_vec, dtype=np.float32)

    where_parts: list[sql.Composable] = [
        sql.SQL("model_id = {} AND bands = {}").format(
            sql.Literal(model_id), sql.Literal(bands)
        )
    ]
    if bbox_26916 is not None:
        xmin, ymin, xmax, ymax = bbox_26916
        where_parts.append(
            sql.SQL(
                "ST_Intersects(footprint, ST_MakeEnvelope({}, {}, {}, {}, 26916))"
            ).format(
                sql.Literal(xmin), sql.Literal(ymin),
                sql.Literal(xmax), sql.Literal(ymax),
            )
        )
    if near_26916 is not None:
        nx, ny, radius_m = near_26916
        where_parts.append(
            sql.SQL(
                "ST_DWithin(center_26916, "
                "ST_SetSRID(ST_MakePoint({}, {}), 26916), {})"
            ).format(sql.Literal(nx), sql.Literal(ny), sql.Literal(radius_m))
        )
    if inside_aoi_only:
        where_parts.append(sql.SQL("inside_aoi"))

    where_clause = sql.SQL(" AND ").join(where_parts)

    # %(q)s and %(top_k)s / %(overfetch)s are still parameter binds.
    # the WHERE composable is built from typed literals (psycopg quotes them).
    query = sql.SQL("""
        WITH ranked AS (
            SELECT chip_location_id,
                   source_cycle,
                   footprint,
                   center_26916,
                   inside_aoi,
                   1 - (embedding <=> %(q)s) AS score
            FROM chip_embeddings
            WHERE {where}
            ORDER BY embedding <=> %(q)s
            LIMIT %(overfetch)s
        )
        SELECT chip_location_id,
               MAX(score) AS max_score,
               (array_agg(source_cycle ORDER BY score DESC))[1] AS winning_cycle,
               (array_agg(ST_X(center_26916) ORDER BY score DESC))[1] AS cx,
               (array_agg(ST_Y(center_26916) ORDER BY score DESC))[1] AS cy,
               (array_agg(ST_XMin(footprint) ORDER BY score DESC))[1] AS xmin,
               (array_agg(ST_YMin(footprint) ORDER BY score DESC))[1] AS ymin,
               (array_agg(ST_XMax(footprint) ORDER BY score DESC))[1] AS xmax,
               (array_agg(ST_YMax(footprint) ORDER BY score DESC))[1] AS ymax,
               bool_or(inside_aoi) AS any_inside_aoi
        FROM ranked
        GROUP BY chip_location_id
        ORDER BY max_score DESC
        LIMIT %(top_k)s
    """).format(where=where_clause)

    ef_search = _resolve_ef_search()

    def _run(c: psycopg.Connection) -> list[tuple]:
        with c.cursor() as cur:
            # session-level SET so it applies even on autocommit connections
            cur.execute(sql.SQL("SET hnsw.ef_search = {}").format(
                sql.Literal(ef_search)
            ))
            cur.execute(query, {"q": q, "overfetch": overfetch, "top_k": top_k})
            return cur.fetchall()

    if conn is None:
        with connect() as own:
            rows = _run(own)
    else:
        rows = _run(conn)

    return [
        SearchHit(
            chip_location_id=loc,
            score=float(score),
            winning_cycle=str(cyc),
            center_26916=(float(cx), float(cy)),
            bbox_26916=(float(xmin), float(ymin), float(xmax), float(ymax)),
            inside_aoi=bool(any_in),
        )
        for (loc, score, cyc, cx, cy, xmin, ymin, xmax, ymax, any_in) in rows
    ]
