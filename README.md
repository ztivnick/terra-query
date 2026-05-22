# terra-query

Natural-language retrieval of unmapped features in public forest. Type
"waterfall" or "old cabin" and get a map of likely candidates ranked by
semantic similarity, not by what's already in OSM.

Status: MVP / POC. Current target area is the Bond Falls block of
Ottawa National Forest (~25 km^2). Production scaling target is ~500 km^2
(Bond Falls + adjacent watersheds).

## Layout

| top-level | what's there |
|---|---|
| [src/terra_query/](src/terra_query/) | the Python package (src-layout, snake_case import name) |
| [frontend/](frontend/) | the web UI slot |
| [data/](data/) | all artifacts; see [data/README.md](data/README.md) for the four-bucket layout |
| [tests/](tests/) | pytest suite; mirrors the package layout |
| `pyproject.toml`, `uv.lock` | Python deps, locked via [uv](https://docs.astral.sh/uv/) |
| `.gitignore` | what isn't committed, with regen pointers inline |

The repo name is `terra-query` (kebab-case, the project / git
convention); the Python package is `terra_query` (snake_case, the
import-name convention; hyphens are illegal in Python identifiers). The
`src/` wrapper is the PyPA src-layout, which prevents accidental
imports from the project root.

## Fresh-clone quickstart

```bash
# 1. install / update Python deps (lockfile-driven; reproducible)
uv sync

# 2. run the tests. Tests that depend on bulk data (NAIP COGs, model
#    weights, embeddings) skip cleanly with a hint pointing at the CLI
#    that produces the missing artifact.
uv run pytest -q
```

Now you have a working clone of all *code* and *small ground-truth /
verification artifacts*. The bulk data (downloaded rasters, model
weights, embeddings, vector stores) is intentionally not in the repo;
the next subsection regenerates it.

## Configuring an experiment

The single source of truth for every "what to run" knob is the
experiment YAML at [experiments/bond_falls_25km_poc.yaml](experiments/bond_falls_25km_poc.yaml).
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
| `cycles` | NAIP cycle ids in the embedding sweep |

Resolution order for which YAML gets loaded: explicit `--experiment <path>`
arg -> `TERRA_QUERY_EXPERIMENT_CONFIG` env var -> the default file above.
Every CLI accepts `--experiment <path>` if you want to point at a different
config.

The working CRS (currently `EPSG:26916`) and the DSN
(`TERRA_QUERY_DATABASE_URL`) stay outside the YAML by design: CRS is
baked into the DB schema's `geometry(..., 26916)` column types, so
changing it requires a schema regeneration + destructive table reset
that isn't a per-experiment knob. DSN is a 12-factor
secret/deployment knob, kept as an env var so credentials stay out
of the checked-in YAML.

## Running everything

```bash
# Bring up the vector store once.
docker compose up -d

# Then: edit experiments/bond_falls_25km_poc.yaml, run one command.
uv run python -m terra_query.cli.run_experiment
```

`run_experiment` is the sequential umbrella over every stage. Each
stage is already idempotent, so a no-op re-run after no YAML change
is cheap. The 10 stages, in order:

1. `reproject_inputs` — AOI + eval set into the working CRS
2. `discover_aerial` — STAC pick: writes NAIP + S2 manifests
3. `fetch_naip` — per-cycle COG download
4. `fetch_sentinel2` — per-scene COG download
5. `build_chip_index` — chip grid + eval-chip PNGs + grid overview
6. `fetch_weights` — model weights download
7. `embed_chips` — sweep over bands x cycles
8. `init_db` — apply schema (idempotent CREATE IF NOT EXISTS)
9. `load_embeddings` — upsert into chip_embeddings
10. `run_n0_retrieval` — regenerate the gate report

```bash
# preview only
uv run python -m terra_query.cli.run_experiment --dry-run

# subset
uv run python -m terra_query.cli.run_experiment --from embed_chips
uv run python -m terra_query.cli.run_experiment --only build_chip_index
```

Each stage can also be invoked directly with its own `--experiment` arg:

```bash
uv run python -m terra_query.ingest.cli.reproject_inputs
uv run python -m terra_query.ingest.cli.discover_aerial
uv run python -m terra_query.ingest.cli.fetch_naip
uv run python -m terra_query.ingest.cli.fetch_sentinel2
uv run python -m terra_query.eval.review            # eval-feature NAIP chips + AOI overlay
uv run python -m terra_query.eval.gate_visuals      # cross-cycle / S2 RGB / dam-on-dam
uv run python -m terra_query.ingest.cli.build_chip_index
uv run python -m terra_query.embed.cli.fetch_weights
uv run python -m terra_query.embed.cli.embed_chips
uv run python -m terra_query.vector_store.cli.init_db
uv run python -m terra_query.vector_store.cli.load_embeddings
uv run python -m terra_query.eval.cli.run_n0_retrieval
```

Every fetch CLI is idempotent: rerunning skips already-valid outputs
and just refreshes manifest metadata. First NAIP fetch is the only
slow step (~10 min/cycle on a residential connection, six cycles).

## Production model (aerial branch)

Locked to **GeoRSCLIP ViT-L/14-336** on RGB. The experiment YAML's
`model_id` is what every CLI reads; the `PRODUCTION_MODEL_ID` constant
in [src/terra_query/embed/models.py](src/terra_query/embed/models.py)
is the registered default for callers without a config (the in-memory
`evaluate_concept` test fixture).

LiDAR / sub-canopy detection is a separate modality and will register
its own model when that branch of the pipeline is built.

### Swapping the production model

Two ways to use the swap CLI; both go through the same code path.

```bash
# (A) edit the YAML by hand, then run with no args
vim experiments/bond_falls_25km_poc.yaml   # bump model_id
uv run python -m terra_query.embed.cli.swap_model

# (B) explicit override; on success, the YAML is rewritten to point at --to
uv run python -m terra_query.embed.cli.swap_model --to <candidate-id> --dry-run
uv run python -m terra_query.embed.cli.swap_model --to <candidate-id> --yes
```

The candidate must already be registered in
[src/terra_query/embed/models.py](src/terra_query/embed/models.py)
`MODELS{}` — add a `ModelSpec` entry for it first.

`swap_model` orchestrates: fetch weights -> embed chips -> load into
chip_embeddings -> regenerate the N0 gate -> purge disk + DB artifacts
of the displaced model -> (in `--to` mode) rewrite the YAML `model_id`.
Destructive operations are always behind a confirm prompt or `--yes`.
`--dry-run` prints the plan with no side effects.

## Vector store (Postgres + PostGIS + pgvector)

The production retrieval layer is Postgres with PostGIS and pgvector.
The MVP runs it locally via Docker Compose; deploying to a managed
Postgres is a DSN change with no code change.

### Local setup

```bash
# Bring up Postgres 16 + PostGIS 3.6 + pgvector 0.8 on 127.0.0.1:5433.
# The Dockerfile under infra/postgres/ extends pgvector/pgvector:pg16
# with postgresql-16-postgis-3; both extensions are CREATEd on the
# initial cluster bring-up via infra/postgres/initdb.d/01_extensions.sql.
docker compose up -d

# Apply the schema (table + HNSW + GIST indexes). Idempotent.
uv run python -m terra_query.vector_store.cli.init_db

# Load embeddings into chip_embeddings. Default: production model on
# RGB across every cycle in chip_index.json. Idempotent
# (ON CONFLICT DO UPDATE on the primary key).
uv run python -m terra_query.vector_store.cli.load_embeddings
```

The DSN is read from the `TERRA_QUERY_DATABASE_URL` env var, defaulting
to `postgresql://terra:terra@localhost:5433/terra_query` (the
docker-compose credentials). Production deploys override the env var.
The DSN intentionally stays out of the experiment YAML: 12-factor is
the standard for secrets, the YAML is checked in (credentials would
leak), and the deploy-target choice is orthogonal to the experiment.

### Schema

| column | type | notes |
|---|---|---|
| `model_id` | TEXT | part of PK; supports multiple models without code change |
| `bands` | TEXT | part of PK; `rgb` today |
| `chip_id` | TEXT | part of PK; e.g. `naip_2022_r034_c028` |
| `chip_location_id` | TEXT | year-stripped grid id; e.g. `r034_c028`; shared across cycles |
| `source_cycle` | TEXT | e.g. `2022` |
| `inside_aoi` | BOOLEAN | true for the 1,822 of 2,209 chips inside the AOI proper |
| `embedding` | `vector(768)` | HNSW index, `vector_cosine_ops` |
| `footprint` | `geometry(Polygon, 26916)` | GIST index |
| `center_26916` | `geometry(Point, 26916)` | GIST index |

Search SQL (see `src/terra_query/vector_store/search.py`) overfetches
via HNSW, GROUPs BY `chip_location_id`, and takes `MAX(score)` per
location so different cycles for the same place don't fight. Spatial
filters (`ST_Intersects(footprint, ST_MakeEnvelope(...))`,
`ST_DWithin(center_26916, ...)`) compose with the ANN.

### Resetting the store

```bash
docker compose down -v   # wipes the named pgdata volume
docker compose up -d
uv run python -m terra_query.vector_store.cli.init_db
uv run python -m terra_query.vector_store.cli.load_embeddings
```

### Integration tests

`tests/vector_store/*_integration.py` and `tests/eval/test_n0_retrieval_db.py`
talk to a real Postgres. They read the DSN from `TEST_DATABASE_URL`,
falling back to `TERRA_QUERY_DATABASE_URL`, falling back to the
docker-compose default. Each session creates one ephemeral schema and
tears it down at teardown, so they don't touch production data even
when pointed at the same instance. If no DB is reachable, the
integration tests skip cleanly.

## What isn't git tracked

**Not tracked** (intentional; regenerable):

- Bulk rasters (`*.tif`, `*.las`, etc.) - regen via ingest CLIs above.
- Embeddings, vector stores, derived arrays (`*.npy`, `*.parquet`,
  `*.faiss`, ...) - regen via the pipeline step that produces them.
- ML model weights (`*.safetensors`, `*.pt`, `*.onnx`, ...) - refetched
  from their source registry at the step that uses them.
- The Python `.venv/`, caches, and lock-of-locks - regen via `uv sync`.
- `frontend/node_modules/` and friends (once the frontend exists) -
  regen via the frontend's package manager.

## Where things go (for contributors)

Path conventions are documented in two places:

- [src/terra_query/core/paths.py](src/terra_query/core/paths.py) is
  the canonical source of every data path the code reads or writes.
  No other module hardcodes a `data/` path. Any layout change starts
  and ends here.
- [data/README.md](data/README.md) explains the four-bucket model
  (`human_authored/`, `source_downloads/`, `pipeline_outputs/`,
  `verification/`) used inside `data/`.

If you're about to write or read a new file, add or use a constant in
`core/paths.py` first.
