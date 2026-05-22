"""Pydantic request / response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_id: str
    embed_dim: int
    experiment_id: str


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    inside_aoi_only: bool = False


class SearchHitOut(BaseModel):
    chip_location_id: str
    score: float
    winning_cycle: str
    center_wgs84: tuple[float, float]  # (lat, lon)
    bbox_wgs84: list[tuple[float, float]]  # closed ring of (lat, lon)
    inside_aoi: bool
    thumbnail_url: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    inside_aoi_only: bool
    results: list[SearchHitOut]


class AoiResponse(BaseModel):
    aoi_id: str
    geojson: dict[str, Any]
