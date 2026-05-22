"""Experiment-config loader.

One YAML per experiment under `experiments/<id>.yaml`. Loaded as a
plain dict; no Pydantic / JSON Schema validation at R1. R2 lifts
this to a typed model.

Resolution order for the config path:
1. Explicit `path` arg to `load_experiment()`.
2. `TERRA_QUERY_EXPERIMENT_CONFIG` env var.
3. `<repo>/experiments/bond_falls_25km_poc.yaml` (the MVP default).

If none of these point at an existing file, `load_experiment` raises
`SystemExit` with a message listing what it tried.

The accessor helpers (`experiment_id_of`, `model_id_of`, ...) fail
loudly via `KeyError` when a required key is missing. Don't bury
defaults here; defaults belong in the YAML.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from terra_query.core.paths import EXPERIMENTS_DIR

CONFIG_ENV_VAR = "TERRA_QUERY_EXPERIMENT_CONFIG"
DEFAULT_CONFIG_PATH = EXPERIMENTS_DIR / "bond_falls_25km_poc.yaml"

REQUIRED_KEYS = (
    "experiment_id",
    "aoi_id",
    "eval_set_id",
    "model_id",
    "modality",
    "bands",
    "chip_params",
    "cycles",
)


def _resolve_path(path: Path | str | None) -> Path:
    """Return the config path to read, following arg -> env -> default."""
    if path is not None:
        return Path(path)
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_experiment(path: Path | str | None = None) -> dict[str, Any]:
    """Load the experiment YAML and return it as a plain dict.

    Validates only that every key in `REQUIRED_KEYS` is present. No
    type checking; that's R2's job.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        tried = [
            f"  arg               : {path!r}",
            f"  ${CONFIG_ENV_VAR} : {os.environ.get(CONFIG_ENV_VAR)!r}",
            f"  default           : {DEFAULT_CONFIG_PATH}",
        ]
        raise SystemExit(
            f"experiment config not found at {resolved}.\n"
            "tried:\n" + "\n".join(tried)
        )

    with resolved.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"experiment config at {resolved} is not a YAML mapping; "
            f"got {type(cfg).__name__}"
        )

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(
            f"experiment config at {resolved} is missing required keys: "
            f"{missing}. expected all of {list(REQUIRED_KEYS)}."
        )

    cfg["_source_path"] = str(resolved)
    return cfg


# ---- accessors (no buried defaults; missing key -> KeyError) ----

def experiment_id_of(cfg: dict) -> str:
    return cfg["experiment_id"]


def aoi_id_of(cfg: dict) -> str:
    return cfg["aoi_id"]


def eval_set_id_of(cfg: dict) -> str:
    return cfg["eval_set_id"]


def model_id_of(cfg: dict) -> str:
    return cfg["model_id"]


def modality_of(cfg: dict) -> str:
    return cfg["modality"]


def bands_of(cfg: dict) -> str:
    return cfg["bands"]


def cycles_of(cfg: dict) -> list[str]:
    # tolerate ints in the YAML, but the canonical type is str (matches
    # NAIP manifest year keys + filename segments downstream)
    return [str(c) for c in cfg["cycles"]]


def chip_params_of(cfg: dict) -> dict[str, int]:
    cp = cfg["chip_params"]
    return {"chip_size_m": int(cp["chip_size_m"]), "stride_m": int(cp["stride_m"])}


def source_path_of(cfg: dict) -> Path:
    """Return the YAML path the config was loaded from (for `swap_model`'s rewrite)."""
    return Path(cfg["_source_path"])
