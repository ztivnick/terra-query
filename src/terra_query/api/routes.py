"""HTTP routes."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from terra_query.api.dependencies import get_db_conn
from terra_query.api.schemas import (
    AoiResponse,
    HealthResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
)
from terra_query.core.crs import to_wgs84
from terra_query.core.paths import aoi_wgs84, chip_thumbnail_key

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    s = request.app.state
    return HealthResponse(
        status="ok",
        model_id=s.model_id,
        embed_dim=s.embed_dim,
        experiment_id=s.experiment_id,
    )


@router.get("/aoi", response_model=AoiResponse)
def aoi(request: Request) -> AoiResponse:
    aoi_id = request.app.state.aoi_id
    path = aoi_wgs84(aoi_id)
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"AOI file missing: {path}")
    geojson = json.loads(path.read_text())
    return AoiResponse(aoi_id=aoi_id, geojson=geojson)


# generic aerial-context templates; used as a fallback when the user's
# query doesn't match a registered eval concept.
_PROMPT_TEMPLATES = (
    "an aerial photo of {text}",
    "an aerial photo of a {text}",
    "an aerial view of {text}",
    "an aerial view of a {text}",
    "satellite imagery of {text}",
    "satellite imagery of a {text}",
    "an overhead photo of {text}",
    "{text}",
)


def _ensemble_prompts(text: str) -> list[str]:
    """If the text matches a registered concept, return its curated prompt
    ensemble; otherwise wrap the text in generic aerial-context templates."""
    from terra_query.eval.queries import all_concepts

    stripped = text.strip()
    norm = stripped.lower().replace(" ", "_")
    concepts = all_concepts()
    if norm in concepts:
        return concepts[norm]
    return [t.format(text=stripped) for t in _PROMPT_TEMPLATES]


def _xy_to_latlon(xy_to_wgs: Any, x: float, y: float) -> tuple[float, float]:
    # pyproj always_xy returns (lon, lat); flip for the frontend.
    lon, lat = xy_to_wgs.transform(x, y)
    return (lat, lon)


def _bbox_to_ring_wgs84(
    xy_to_wgs: Any, xmin: float, ymin: float, xmax: float, ymax: float
) -> list[tuple[float, float]]:
    corners_xy = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ]
    return [_xy_to_latlon(xy_to_wgs, x, y) for x, y in corners_xy]


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    request: Request,
    conn=Depends(get_db_conn),
) -> SearchResponse:
    s = request.app.state
    if s.model is None:
        raise HTTPException(
            status_code=503, detail="model not loaded (TERRA_QUERY_SKIP_MODEL_LOAD=1)"
        )

    from terra_query.embed.query import encode_prompt_ensemble
    from terra_query.vector_store.search import search as vs_search

    prompts = _ensemble_prompts(body.text)
    text_vec = encode_prompt_ensemble(prompts, s.model, s.tokenizer, s.device)
    text_vec = np.asarray(text_vec, dtype=np.float32)

    hits = vs_search(
        query_vec=text_vec,
        top_k=body.top_k,
        inside_aoi_only=body.inside_aoi_only,
        model_id=s.model_id,
        bands=s.bands,
        conn=conn,
    )

    xy_to_wgs = to_wgs84()
    out: list[SearchHitOut] = []
    for h in hits:
        cx, cy = h.center_26916
        xmin, ymin, xmax, ymax = h.bbox_26916
        center_latlon = _xy_to_latlon(xy_to_wgs, cx, cy)
        bbox_ring = _bbox_to_ring_wgs84(xy_to_wgs, xmin, ymin, xmax, ymax)
        key = chip_thumbnail_key(h.chip_location_id, h.winning_cycle, s.bands)
        thumb_url = s.storage.url_for(key)
        out.append(
            SearchHitOut(
                chip_location_id=h.chip_location_id,
                score=h.score,
                winning_cycle=h.winning_cycle,
                center_wgs84=center_latlon,
                bbox_wgs84=bbox_ring,
                inside_aoi=h.inside_aoi,
                thumbnail_url=thumb_url,
            )
        )

    return SearchResponse(
        query=body.text,
        top_k=body.top_k,
        inside_aoi_only=body.inside_aoi_only,
        results=out,
    )


@router.get("/thumbnails/{key}")
def thumbnail(key: str, request: Request) -> Response:
    s = request.app.state
    if not key.endswith(".png"):
        raise HTTPException(status_code=400, detail="key must end in .png")

    # key shape: "<chip_location_id>__<cycle>__<bands>.png"
    stem = key[: -len(".png")]
    parts = stem.split("__")
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail=f"bad key shape: {key}")
    chip_location_id, cycle, bands = parts
    if bands != s.bands:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported bands {bands!r}; this api serves {s.bands!r}",
        )

    storage = s.storage
    if storage.exists(key):
        data = storage.read_bytes(key)
        return Response(content=data, media_type="image/png")

    # cache miss: render from the source COG
    from PIL import Image

    from terra_query.core.paths import chip_index_json, naip_cog
    from terra_query.ingest.chips import BAND_TUPLES, ChipBox, read_chip

    if bands not in BAND_TUPLES:
        raise HTTPException(status_code=400, detail=f"unknown bands {bands!r}")

    chip_index = json.loads(chip_index_json(s.experiment_id).read_text())
    chip_block = None
    for c in chip_index["cycles"]:
        if str(c["year"]) == cycle:
            chip_block = c
            break
    if chip_block is None:
        raise HTTPException(status_code=404, detail=f"unknown cycle {cycle!r}")

    chip_row = None
    for ch in chip_block["chips"]:
        if f"r{ch['row']:03d}_c{ch['col']:03d}" == chip_location_id:
            chip_row = ch
            break
    if chip_row is None:
        raise HTTPException(
            status_code=404, detail=f"unknown chip_location_id {chip_location_id!r}"
        )

    cog_path = naip_cog(s.aoi_id, cycle)
    if not cog_path.exists():
        raise HTTPException(status_code=500, detail=f"missing source COG: {cog_path}")

    xmin, ymin, _xmax, _ymax = chip_row["bbox_26916"]
    chip_box = ChipBox(
        row=int(chip_row["row"]),
        col=int(chip_row["col"]),
        west=float(xmin),
        south=float(ymin),
    )
    arr = read_chip(
        chip=chip_box,
        source_cog_path=cog_path,
        bands=BAND_TUPLES[bands],
    )
    # (bands, h, w) -> (h, w, bands) for PIL
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    storage.write_bytes(key, data)
    return Response(content=data, media_type="image/png")
