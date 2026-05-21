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

# Render the aerial-ingest gate verification visuals.
uv run python -m terra_query.eval.gate_visuals

# Cut the chip grid for every NAIP cycle (writes chip_index.json).
uv run python -m terra_query.ingest.cli.build_chip_index

# Fetch the production model weights (~1.6 GB) into model_weights/.
uv run python -m terra_query.embed.cli.fetch_weights

# Embed every chip with the production model across all 6 cycles
# (~6 .npy files; ~6.5 MB each; one overnight run from cold start).
uv run python -m terra_query.embed.cli.embed_chips

# Run the retrieval gate and write the verdict report + thumbnails.
uv run python -m terra_query.eval.cli.run_n0_retrieval
```

Every fetch CLI is idempotent: rerunning skips already-valid outputs
and just refreshes manifest metadata. First NAIP fetch is the only
slow step (~10 min/cycle on a residential connection, six cycles).

## Production model (aerial branch)

Locked to **GeoRSCLIP ViT-L/14-336** on RGB. Single source of truth:
the `PRODUCTION_MODEL_ID` constant in
[src/terra_query/embed/models.py](src/terra_query/embed/models.py). Every
CLI and every test fixture reads from there.

LiDAR / sub-canopy detection is a separate modality and will register
its own model when that branch of the pipeline is built.

### Swapping the production model

The whole pipeline reads `embed.models.PRODUCTION_MODEL_ID`. To swap:

```bash
# 1. register the candidate in src/terra_query/embed/models.py:
#    add a ModelSpec entry to MODELS{} for the new id.

# 2. (optional) validate the candidate against the gate before locking it in:
uv run python -m terra_query.embed.cli.fetch_weights --models <candidate-id>
uv run python -m terra_query.embed.cli.embed_chips --models <candidate-id>
uv run python -m terra_query.eval.cli.run_n0_retrieval \
    --configs <candidate-id>__rgb

# 3. point production at the candidate:
#    edit PRODUCTION_MODEL_ID in src/terra_query/embed/models.py.

# 4. purge the now-stale artifacts of the old production model:
rm -rf data/source_downloads/model_weights/<old-id>
rm data/pipeline_outputs/embeddings/<old-id>__*.npy \
   data/pipeline_outputs/embeddings/<old-id>__*.json
rm -rf data/verification/gate/topk_chips/*<old-id>*
#    (also drop the old MODELS entry from models.py if you're done with it)

# 5. regenerate the gate + thumbnails with the new production model:
uv run python -m terra_query.eval.cli.run_n0_retrieval
```

This is a manual procedure today. A future orchestrator step will
replace it with a single `terra-query run` command that walks the
dependency graph and purges stale artifacts automatically.

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
