"""Single source of truth for every data path in the repo.

Every module that touches the filesystem imports from here. The literal
layout under `data/` is not duplicated anywhere else. If the layout
changes, this file is the only place that needs editing.

Four data buckets, organized by provenance (not by file type):

- `human_authored/`: hand-curated, project-defining; never regenerated.
- `source_downloads/`: fetched verbatim from external sources.
- `pipeline_outputs/`: produced by this project's code.
- `verification/`: eyeball-check artifacts and per-run logs.
"""

from __future__ import annotations

from pathlib import Path

# core/paths.py -> core/ -> terra_query/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA = REPO_ROOT / "data"
HUMAN_AUTHORED = DATA / "human_authored"
SOURCE_DOWNLOADS = DATA / "source_downloads"
PIPELINE_OUTPUTS = DATA / "pipeline_outputs"
VERIFICATION = DATA / "verification"

# human_authored/
AOI_WGS84 = HUMAN_AUTHORED / "aoi" / "bond_falls_block.geojson"
EVAL_WGS84 = HUMAN_AUTHORED / "eval" / "known_features.geojson"
EVAL_README = HUMAN_AUTHORED / "eval" / "README.md"

# source_downloads/
NAIP_DIR = SOURCE_DOWNLOADS / "naip"
NAIP_MANIFEST = NAIP_DIR / "manifest.json"
S2_DIR = SOURCE_DOWNLOADS / "sentinel2"
S2_MANIFEST = S2_DIR / "manifest.json"
MODEL_WEIGHTS_DIR = SOURCE_DOWNLOADS / "model_weights"

# pipeline_outputs/
AOI_26916 = PIPELINE_OUTPUTS / "aoi" / "bond_falls_block_26916.geojson"
EVAL_26916 = PIPELINE_OUTPUTS / "eval" / "known_features_26916.geojson"
CHIPS_DIR = PIPELINE_OUTPUTS / "chips"
CHIP_INDEX_JSON = CHIPS_DIR / "chip_index.json"
CHIP_EVAL_DIR = CHIPS_DIR / "eval"

# verification/
CRS_VERIFICATION = VERIFICATION / "crs_verification.txt"
EVAL_CHIPS_DIR = VERIFICATION / "eval_chips"
GATE_DIR = VERIFICATION / "gate"
OVERLAY_PNG = GATE_DIR / "overlay_check.png"
CHIP_GRID_OVERVIEW_PNG = GATE_DIR / "chip_grid_overview.png"
RUN_LOGS_DIR = VERIFICATION / "runs"


def naip_cog(year: str) -> Path:
    return NAIP_DIR / year / f"naip_bondfalls_{year}_26916.tif"


def s2_cog(date_str: str) -> Path:
    return S2_DIR / date_str / f"s2_bondfalls_{date_str}_26916.tif"


def eval_chip(feature_id: str) -> Path:
    return EVAL_CHIPS_DIR / f"{feature_id}.png"


def chip_eval(feature_id: str) -> Path:
    return CHIP_EVAL_DIR / f"{feature_id}.png"
