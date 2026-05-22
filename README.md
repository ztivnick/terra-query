# terra-query

Natural-language retrieval of unmapped features in public forest. Type
"waterfall" or "old cabin" and get a map of likely candidates ranked
by semantic similarity, not by what's already in OSM.

Status: MVP / POC. Current AOI is the Bond Falls block of Ottawa
National Forest (~25 km^2); production scaling target is ~500 km^2
(Bond Falls + adjacent watersheds). Active config:
[experiments/bond_falls_25km_poc.yaml](experiments/bond_falls_25km_poc.yaml).
Current model: GeoRSCLIP ViT-L/14-336 on RGB. AOI and model are
config-driven.

The repo has two surfaces:

- **the pipeline** - offline stages under
  `src/terra_query/{ingest,embed,vector_store,eval}/` that fetch
  imagery, build the chip grid, embed, load Postgres, and produce the
  eval gate report.
- **the app** - FastAPI backend under `src/terra_query/api/` plus a
  Vite/Leaflet frontend under `frontend/`, launched by `./start`.

Both talk to the same Postgres and the same model weights. The
pipeline writes; the app reads.

## Layout

| top-level | what's there |
|---|---|
| [src/terra_query/](src/terra_query/) | the Python package |
| [frontend/](frontend/) | the web UI (Vite + TypeScript + Tailwind + Leaflet) |
| [data/](data/) | all artifacts; layout in [data/README.md](data/README.md) |
| [experiments/](experiments/) | per-experiment YAML configs |
| [infra/](infra/) | Dockerfiles for the db + api containers |
| [tests/](tests/) | pytest suite; mirrors the package layout |
| `start` | launches the local stack |
| `docker-compose.yml` | the db + api services |
| `pyproject.toml`, `uv.lock` | Python deps, locked via [uv](https://docs.astral.sh/uv/) |

Project name is `terra-query`; Python package is `terra_query`
(hyphens are illegal in identifiers). `src/` is the PyPA src-layout.

## Setup

Prereqs on `PATH`:

- [`uv`](https://docs.astral.sh/uv/) - Python toolchain
- `docker` - for the Postgres + pgvector container
- `npm` (Node 22+) - for the frontend

```bash
uv sync
uv run pytest -q
```

Bulk data (rasters, model weights, embeddings, vector store) is not
in the repo. The pipeline below regenerates it.

---

## The pipeline

One YAML at
[experiments/bond_falls_25km_poc.yaml](experiments/bond_falls_25km_poc.yaml).
Eight keys:

| key | what it controls |
|---|---|
| `experiment_id` | namespace for `data/pipeline_outputs/<id>/` + `data/verification/<id>/` |
| `aoi_id` | resolves to `data/human_authored/aoi/<id>.geojson` |
| `eval_set_id` | resolves to `data/human_authored/eval/<id>.geojson` |
| `model_id` | a key in [src/terra_query/embed/models.py](src/terra_query/embed/models.py) `MODELS{}` |
| `modality` | informational today; aerial is the only modality the pipeline reads |
| `bands` | `rgb` or `cir` |
| `chip_params` | `chip_size_m`, `stride_m` (chip grid geometry) |
| `cycles` | imagery cycle ids in the embedding sweep |

Config resolution: `--experiment <path>` arg, then
`TERRA_QUERY_EXPERIMENT_CONFIG` env var, then the default file above.
Every CLI accepts `--experiment <path>`.

Working CRS (`EPSG:26916`) and DSN (`TERRA_QUERY_DATABASE_URL`) live
outside the YAML. CRS is baked into the DB schema's
`geometry(..., 26916)` column types; changing it forces a destructive
table reset. DSN is a 12-factor secret.

### Stages

```bash
docker compose up -d db
uv run python -m terra_query.cli.run_experiment
```

`run_experiment` walks the ten stages in order. Each stage is
idempotent.

1. `reproject_inputs` - AOI + eval set into the working CRS
2. `discover_aerial` - STAC pick; writes per-source manifests
3. `fetch_naip` - per-cycle NAIP COG download
4. `fetch_sentinel2` - per-scene Sentinel-2 COG download
5. `build_chip_index` - chip grid + eval-chip PNGs + grid overview
6. `fetch_weights` - model weights download
7. `embed_chips` - sweep over bands x cycles
8. `init_db` - apply schema (idempotent `CREATE IF NOT EXISTS`)
9. `load_embeddings` - upsert into `chip_embeddings`
10. `run_n0_retrieval` - regenerate the gate report

Subset flags: `--dry-run`, `--from <stage>`, `--to <stage>`,
`--only <stage>`.

Each stage is also invokable directly:

```bash
uv run python -m terra_query.ingest.cli.reproject_inputs
uv run python -m terra_query.ingest.cli.discover_aerial
uv run python -m terra_query.ingest.cli.fetch_naip
uv run python -m terra_query.ingest.cli.fetch_sentinel2
uv run python -m terra_query.ingest.cli.build_chip_index
uv run python -m terra_query.embed.cli.fetch_weights
uv run python -m terra_query.embed.cli.embed_chips
uv run python -m terra_query.vector_store.cli.init_db
uv run python -m terra_query.vector_store.cli.load_embeddings
uv run python -m terra_query.eval.cli.run_n0_retrieval
uv run python -m terra_query.eval.review            # eval-feature NAIP chips + AOI overlay
uv run python -m terra_query.eval.gate_visuals      # cross-cycle / S2 RGB / dam-on-dam
```

First NAIP fetch takes ~10 min/cycle on a residential connection;
the rest of the pipeline runs in well under a minute.

Gate report:
`data/verification/<experiment_id>/gate/n0_retrieval_report.md`.

### Swapping the model

```bash
vim experiments/bond_falls_25km_poc.yaml   # update model_id
uv run python -m terra_query.embed.cli.swap_model
```

`swap_model` chains fetch -> embed -> load -> regen gate -> purge the
displaced model's artifacts. Flags: `--to <id>` overrides the YAML
and rewrites it on success; `--dry-run` previews; `--yes` skips the
confirm. Candidates must be registered as a `ModelSpec` in
[src/terra_query/embed/models.py](src/terra_query/embed/models.py)
`MODELS{}`.

`PRODUCTION_MODEL_ID` in `embed/models.py` is the registered default
for callers without a config (the in-memory test fixture); the YAML
supersedes it everywhere else.

---

## The app

```bash
./start
```

Spawns three processes:

- Postgres via docker-compose
- uvicorn on <http://localhost:8000> (host-bound; MPS available)
- Vite dev server on <http://localhost:5173> (HMR)

Log lines from all three interleave in the terminal with colored
per-source prefixes. Ctrl+C, SIGTERM, or terminal close (SIGHUP)
tears down all three including grandchildren. A second `./start`
takes over from the first via a pidfile in `/tmp`; a stale `vite` or
`uvicorn` from a previous crash is reclaimed before spawning.

### Backend

FastAPI. The production model loads once at startup into `app.state`.

| method | path | what it returns |
|---|---|---|
| GET | `/healthz` | `model_id`, `embed_dim`, `experiment_id` |
| GET | `/aoi` | WGS84 GeoJSON of the configured AOI |
| POST | `/search` | text -> top-K chip-locations + footprints + thumbnail URLs |
| GET | `/thumbnails/{key}` | per-chip PNG, rendered on demand and cached on disk |

Env config:

| env var | what it controls |
|---|---|
| `TERRA_QUERY_DATABASE_URL` | psycopg DSN; defaults to the docker-compose creds |
| `TERRA_QUERY_CORS_ORIGINS` | comma-separated allowlist; defaults to Vite dev + preview origins |
| `TERRA_QUERY_DEVICE` | `mps` / `cuda` / `cpu`; auto-detects if unset |
| `TERRA_QUERY_EXPERIMENT_CONFIG` | path to the experiment YAML |
| `TERRA_QUERY_SKIP_MODEL_LOAD` | `1` skips the weights load (test fixture only) |

The docker-compose `api` service builds a CPU-only image of the
backend for hosts without MPS.

### Frontend

Vite + TypeScript + Tailwind v4 + Leaflet. `npm run build` emits a
static `dist/` of HTML / CSS / JS. Base map: Esri World Imagery +
Esri Boundaries-and-Places transparent labels overlay; no keys. All
URLs env-driven: `VITE_TERRA_QUERY_API_URL`,
`VITE_TERRA_QUERY_TILE_URL`, `VITE_TERRA_QUERY_LABELS_URL`. Build,
run, and configuration details in
[frontend/README.md](frontend/README.md).

---

## Vector store

Postgres + PostGIS + pgvector. One docker-compose service locally:

```bash
docker compose up -d db
uv run python -m terra_query.vector_store.cli.init_db
uv run python -m terra_query.vector_store.cli.load_embeddings
```

DSN: `TERRA_QUERY_DATABASE_URL`, default
`postgresql://terra:terra@localhost:5433/terra_query`.

### Schema

| column | type | notes |
|---|---|---|
| `model_id` | TEXT | part of PK; multiple models without code change |
| `bands` | TEXT | part of PK |
| `chip_id` | TEXT | part of PK; e.g. `naip_2022_r034_c028` |
| `chip_location_id` | TEXT | year-stripped grid id; shared across cycles |
| `source_cycle` | TEXT | the cycle the chip was read from |
| `inside_aoi` | BOOLEAN | true for chips whose center sits inside the AOI polygon |
| `embedding` | `vector(768)` | HNSW index, `vector_cosine_ops` |
| `footprint` | `geometry(Polygon, 26916)` | GIST index |
| `center_26916` | `geometry(Point, 26916)` | GIST index |

Search SQL
([src/terra_query/vector_store/search.py](src/terra_query/vector_store/search.py))
overfetches via HNSW, groups by `chip_location_id`, and takes
`MAX(score)` per location so different cycles for the same place
don't fight. Spatial filters compose with the ANN. HNSW `ef_search`
is bumped from pgvector's default (40) to 1000;
`TERRA_QUERY_HNSW_EF_SEARCH` overrides.

### Reset

```bash
docker compose down -v
docker compose up -d db
uv run python -m terra_query.vector_store.cli.init_db
uv run python -m terra_query.vector_store.cli.load_embeddings
```

### Integration tests

`tests/vector_store/*_integration.py` and
`tests/eval/test_n0_retrieval_db.py` hit a real Postgres. DSN
resolution: `TEST_DATABASE_URL` -> `TERRA_QUERY_DATABASE_URL` ->
docker-compose default. Each session uses an ephemeral schema.
Skipped if no DB is reachable.

---

## Paths

- [src/terra_query/core/paths.py](src/terra_query/core/paths.py) -
  every data path the code reads or writes. No other module
  hardcodes paths under `data/`.
- [data/README.md](data/README.md) - the four-bucket layout
  (`human_authored/`, `source_downloads/`, `pipeline_outputs/`,
  `verification/`).
