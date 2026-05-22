"""FastAPI app factory + lifespan + CORS.

Env vars:
    TERRA_QUERY_EXPERIMENT_CONFIG  path to the experiment YAML
    TERRA_QUERY_DATABASE_URL       psycopg DSN
    TERRA_QUERY_CORS_ORIGINS       comma-separated allowlist
    TERRA_QUERY_DEVICE             "mps" | "cuda" | "cpu" (auto if unset)
    TERRA_QUERY_SKIP_MODEL_LOAD    "1" to skip the model load (tests)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from terra_query.api.routes import router
from terra_query.core import config as cfg_mod
from terra_query.core.paths import chip_thumbnails_dir
from terra_query.core.storage import LocalFilesystemStorage

CORS_ORIGINS_ENV = "TERRA_QUERY_CORS_ORIGINS"
DEVICE_ENV = "TERRA_QUERY_DEVICE"
SKIP_MODEL_ENV = "TERRA_QUERY_SKIP_MODEL_LOAD"
THUMBNAILS_URL_PREFIX = "/thumbnails/"

DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://localhost:4173")


def _resolve_cors_origins() -> list[str]:
    raw = os.environ.get(CORS_ORIGINS_ENV)
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def _resolve_device() -> str:
    explicit = os.environ.get(DEVICE_ENV)
    if explicit:
        return explicit
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(cfg: dict):
    from terra_query.core.paths import MODEL_WEIGHTS_MANIFEST
    from terra_query.embed import models as model_registry

    model_id = cfg_mod.model_id_of(cfg)
    manifest = json.loads(MODEL_WEIGHTS_MANIFEST.read_text())
    entry = manifest["models"][model_id]
    device = _resolve_device()
    model, preprocess, tokenizer = model_registry.load(
        model_id, weights_path=entry.get("weights_path"), device=device
    )
    spec = model_registry.spec(model_id)
    return {
        "model_id": model_id,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "device": device,
        "embed_dim": spec.embed_dim,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = cfg_mod.load_experiment()
    app.state.cfg = cfg
    app.state.experiment_id = cfg_mod.experiment_id_of(cfg)
    app.state.aoi_id = cfg_mod.aoi_id_of(cfg)
    app.state.bands = cfg_mod.bands_of(cfg)

    storage_root = chip_thumbnails_dir(app.state.experiment_id)
    app.state.storage = LocalFilesystemStorage(
        root=storage_root, url_prefix=THUMBNAILS_URL_PREFIX
    )

    if os.environ.get(SKIP_MODEL_ENV) == "1":
        # tests that don't touch /search; skips the weights load
        from terra_query.embed.models import spec as model_spec

        s = model_spec(cfg_mod.model_id_of(cfg))
        app.state.model_id = s.model_id
        app.state.model = None
        app.state.preprocess = None
        app.state.tokenizer = None
        app.state.device = "cpu"
        app.state.embed_dim = s.embed_dim
    else:
        loaded = _load_model(cfg)
        app.state.model_id = loaded["model_id"]
        app.state.model = loaded["model"]
        app.state.preprocess = loaded["preprocess"]
        app.state.tokenizer = loaded["tokenizer"]
        app.state.device = loaded["device"]
        app.state.embed_dim = loaded["embed_dim"]

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="terra-query",
        version="0.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )
    app.include_router(router)
    return app
