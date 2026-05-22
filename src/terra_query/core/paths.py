"""Single source of truth for every data path in the repo.

Every module that touches the filesystem imports from here. The literal
layout under `data/` is not duplicated anywhere else; if the layout
changes, this file is the only place that needs editing.

Four data buckets, organized by provenance (not by file type):

- `human_authored/`: hand-curated, project-defining; never regenerated.
- `source_downloads/`: fetched verbatim from external sources.
- `pipeline_outputs/`: produced by this project's code.
- `verification/`: eyeball-check artifacts and per-run logs.

R1 scoping rules:
- Per-experiment artifacts namespace under
  `data/{pipeline_outputs,verification}/<experiment_id>/`.
- Per-AOI source downloads (NAIP, S2) namespace under
  `data/source_downloads/{naip,sentinel2}/<aoi_id>/`.
- Per-AOI / per-eval-set derivatives sit beside each other under
  `data/pipeline_outputs/{aoi,eval}/<id>_26916.geojson`. The 26916
  suffix encodes the working CRS; R2 generalizes when CRS becomes
  per-experiment.
- Per-model weights stay under `data/source_downloads/model_weights/<model_id>/`.
- The CRS is project-pinned at `core.crs.WORKING_CRS_EPSG`; not in YAML
  at R1. See `r1-orchestrator-scaffold.md` for the R2 plan.

Every function below takes the relevant id(s) as required positional
args (no buried defaults) so every call site visibly declares its
scoping.
"""

from __future__ import annotations

from pathlib import Path

# core/paths.py -> core/ -> terra_query/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "terra_query"

# experiment configs live here, one YAML per experiment
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# checked-in DDL for the vector store
VECTOR_STORE_SCHEMA_SQL = SRC_ROOT / "vector_store" / "schema.sql"

DATA = REPO_ROOT / "data"
HUMAN_AUTHORED = DATA / "human_authored"
SOURCE_DOWNLOADS = DATA / "source_downloads"
PIPELINE_OUTPUTS = DATA / "pipeline_outputs"
VERIFICATION = DATA / "verification"

# checked-in (not per-experiment / per-anything): the README under
# the project-wide eval-set directory, the CRS verification report
# (CRS is project-pinned at R1), per-model weights base dir +
# global manifest, and the catch-all run-logs directory.
EVAL_README = HUMAN_AUTHORED / "eval" / "README.md"
CRS_VERIFICATION = VERIFICATION / "crs_verification.txt"
MODEL_WEIGHTS_DIR = SOURCE_DOWNLOADS / "model_weights"
MODEL_WEIGHTS_MANIFEST = MODEL_WEIGHTS_DIR / "manifest.json"
RUN_LOGS_DIR = VERIFICATION / "runs"


# ---- per-AOI resolvers ----

def aoi_wgs84(aoi_id: str) -> Path:
    """Hand-authored AOI polygon in WGS84."""
    return HUMAN_AUTHORED / "aoi" / f"{aoi_id}.geojson"


def aoi_26916(aoi_id: str) -> Path:
    """AOI reprojected into the working CRS (EPSG:26916).

    The `_26916` suffix encodes the project-pinned working CRS. R2
    generalizes when CRS becomes per-experiment.
    """
    return PIPELINE_OUTPUTS / "aoi" / f"{aoi_id}_26916.geojson"


def naip_aoi_dir(aoi_id: str) -> Path:
    """Root dir for this AOI's NAIP downloads."""
    return SOURCE_DOWNLOADS / "naip" / aoi_id


def naip_manifest(aoi_id: str) -> Path:
    return naip_aoi_dir(aoi_id) / "manifest.json"


def naip_cog(aoi_id: str, year: str) -> Path:
    """The mosaicked NAIP COG for one (AOI, cycle year)."""
    return naip_aoi_dir(aoi_id) / year / f"naip_{aoi_id}_{year}.tif"


def s2_aoi_dir(aoi_id: str) -> Path:
    return SOURCE_DOWNLOADS / "sentinel2" / aoi_id


def s2_manifest(aoi_id: str) -> Path:
    return s2_aoi_dir(aoi_id) / "manifest.json"


def s2_cog(aoi_id: str, date_str: str) -> Path:
    """The stacked-bands Sentinel-2 COG for one (AOI, scene date)."""
    return s2_aoi_dir(aoi_id) / date_str / f"s2_{aoi_id}_{date_str}.tif"


# ---- per-eval-set resolvers ----

def eval_wgs84(eval_set_id: str) -> Path:
    """Hand-authored eval features in WGS84."""
    return HUMAN_AUTHORED / "eval" / f"{eval_set_id}.geojson"


def eval_26916(eval_set_id: str) -> Path:
    """Eval features reprojected into the working CRS."""
    return PIPELINE_OUTPUTS / "eval" / f"{eval_set_id}_26916.geojson"


def eval_chips_dir(eval_set_id: str) -> Path:
    """Per-feature NAIP review chips (rendered by `eval/review.py`)."""
    return VERIFICATION / "eval_chips" / eval_set_id


def eval_chip(eval_set_id: str, feature_id: str) -> Path:
    return eval_chips_dir(eval_set_id) / f"{feature_id}.png"


# ---- per-experiment resolvers ----

def experiment_outputs_dir(experiment_id: str) -> Path:
    return PIPELINE_OUTPUTS / experiment_id


def experiment_verification_dir(experiment_id: str) -> Path:
    return VERIFICATION / experiment_id


def chips_dir(experiment_id: str) -> Path:
    return experiment_outputs_dir(experiment_id) / "chips"


def chip_index_json(experiment_id: str) -> Path:
    return chips_dir(experiment_id) / "chip_index.json"


def chip_eval_dir(experiment_id: str) -> Path:
    """Eval-feature chip PNGs read from the chip grid (vs raw NAIP)."""
    return chips_dir(experiment_id) / "eval"


def chip_eval(experiment_id: str, feature_id: str) -> Path:
    return chip_eval_dir(experiment_id) / f"{feature_id}.png"


def embeddings_dir(experiment_id: str) -> Path:
    return experiment_outputs_dir(experiment_id) / "embeddings"


def embeddings_npy(experiment_id: str, model_id: str, bands: str, cycle: str) -> Path:
    return embeddings_dir(experiment_id) / f"{model_id}__{bands}__{cycle}.npy"


def embeddings_json(experiment_id: str, model_id: str, bands: str, cycle: str) -> Path:
    return embeddings_dir(experiment_id) / f"{model_id}__{bands}__{cycle}.json"


def gate_dir(experiment_id: str) -> Path:
    return experiment_verification_dir(experiment_id) / "gate"


def overlay_png(experiment_id: str) -> Path:
    """Full-AOI NAIP overlay with eval points (review.py).

    Per-experiment for R1 (over-scoped: also depends on AOI + eval set
    + aerial source). Re-scoping into a per-(aoi, eval-set, source)
    layout is R2.
    """
    return gate_dir(experiment_id) / "overlay_check.png"


def chip_grid_overview_png(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "chip_grid_overview.png"


def cross_cycle_png(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "cross_cycle_falls.png"


def dam_on_dam_png(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "dam_on_dam.png"


def s2_rgb_png(experiment_id: str, date_str: str) -> Path:
    return gate_dir(experiment_id) / f"s2_{date_str}_rgb.png"


def n0_results_json(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "n0_retrieval_results.json"


def n0_report_md(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "n0_retrieval_report.md"


def topk_chips_dir(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "topk_chips"


def topk_chip_dir(experiment_id: str, concept: str, model_id: str, bands: str) -> Path:
    return topk_chips_dir(experiment_id) / f"{concept}__{model_id}__{bands}"


def db_roundtrip_log(experiment_id: str) -> Path:
    return gate_dir(experiment_id) / "s06_db_roundtrip.txt"


# ---- per-model resolvers ----

def model_weights_dir(model_id: str) -> Path:
    return MODEL_WEIGHTS_DIR / model_id
