# terra-query

Natural-language retrieval of unmapped features in public forest. Type
"waterfall" or "old cabin" and get a map of likely candidates ranked by
semantic similarity, not by what's already in OSM.

Status: MVP / POC. Current target area is the Bond Falls block of
Ottawa National Forest (~25 km^2). Strategy and per-step plans live in
`docs/architecture_plans/` (kept local by convention, see "What is and
isn't in this clone" below).

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

# 2. verify the install
uv run pytest -q          # 19/19 expected after S3
```

Now you have a working clone of all *code* and *small ground-truth /
verification artifacts*. The bulk data (downloaded rasters, model
weights, embeddings, vector stores) is intentionally not in the repo;
the next subsection regenerates it.

## Regenerating everything that's gitignored

The `.gitignore` is annotated inline with which CLI regenerates each
class of ignored artifact. The full pipeline from an empty `data/` is:

```bash
# Reproject the human-authored AOI + eval set into the working CRS.
# (Usually already-tracked output; rerun if the WGS84 inputs change.)
uv run python -m terra_query.ingest.cli.reproject_inputs

# STAC discovery (no downloads; writes per-source manifests). Network.
uv run python -m terra_query.ingest.cli.discover_aerial

# Fetch NAIP cycles (~1.6 GB total on first run). Idempotent.
uv run python -m terra_query.ingest.cli.fetch_naip

# Fetch Sentinel-2 winter scenes (~5 MB). Idempotent.
uv run python -m terra_query.ingest.cli.fetch_sentinel2

# Render the per-eval-feature NAIP chips + AOI overlay PNG.
uv run python -m terra_query.eval.review

# Render the gate verification visuals.
uv run python -m terra_query.eval.gate_visuals
```

Every fetch CLI is idempotent: rerunning skips already-valid outputs
and just refreshes manifest metadata. First NAIP fetch is the only
slow step (~10 min/cycle on a residential connection, six cycles).

Later pipeline steps (S4 chip cutting, S5 embeddings, S6 vector store,
S7 query path + UI, S8 LiDAR, ...) land their own ingest / regen
commands; each will be documented at its own step.

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
