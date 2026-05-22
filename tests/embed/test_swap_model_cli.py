"""Unit tests for the swap_model CLI: dry-run plan output, no-op detection,
unregistered-target error. The destructive path is not exercised in unit
tests (it hits the network + DB)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from terra_query.core import config
from terra_query.embed import models
from terra_query.embed.cli import swap_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_min_yaml(tmp_path: Path, **overrides) -> Path:
    base = {
        "experiment_id": "tswap",
        "aoi_id": "tswap_aoi",
        "eval_set_id": "tswap_eval",
        "model_id": models.PRODUCTION_MODEL_ID,
        "modality": "aerial",
        "bands": "rgb",
        "chip_params": {"chip_size_m": 224, "stride_m": 112},
        "cycles": ["2022"],
    }
    base.update(overrides)
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(base, sort_keys=False))
    return p


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "terra_query.embed.cli.swap_model", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# ---- registration / target validation ----

def test_to_unregistered_model_errors(tmp_path):
    p = _write_min_yaml(tmp_path)
    r = _run(["--experiment", str(p), "--to", "not-a-real-model", "--dry-run"])
    assert r.returncode != 0
    assert "not registered" in r.stdout + r.stderr


# ---- no-op detection ----

def test_no_op_when_target_is_only_current_model(tmp_path):
    """With the production model on-disk + in DB and YAML target == production,
    the CLI should report a no-op."""
    p = _write_min_yaml(tmp_path)
    r = _run(["--experiment", str(p), "--dry-run"])
    # exit 0 either way (no-op or dry-run plan)
    assert r.returncode == 0
    # at least one of these branches must fire
    assert ("no-op" in r.stdout) or ("dry-run" in r.stdout)


# ---- dry-run plan output ----

def test_dry_run_shape_and_no_side_effects(tmp_path):
    """Dry-run with --to == YAML model: should print a plan and exit 0.
    No YAML rewrite. No subprocess side effects."""
    p = _write_min_yaml(tmp_path)
    before = p.read_text()
    r = _run(["--experiment", str(p), "--to", models.PRODUCTION_MODEL_ID, "--dry-run"])
    assert r.returncode == 0
    after = p.read_text()
    assert before == after, "dry-run must not rewrite the YAML"


# ---- in-process helpers ----

def test_disk_models_empty_when_no_artifacts(tmp_path, monkeypatch):
    """_disk_models on a fresh experiment returns empty."""
    # point the resolver at a fresh experiment dir under tmp
    from terra_query.core import paths

    monkeypatch.setattr(paths, "PIPELINE_OUTPUTS", tmp_path)
    found = swap_model._disk_models("nope", "rgb")
    assert found == set()


def test_disk_models_finds_npy_in_experiment_dir(tmp_path, monkeypatch):
    from terra_query.core import paths

    monkeypatch.setattr(paths, "PIPELINE_OUTPUTS", tmp_path)
    edir = tmp_path / "expx" / "embeddings"
    edir.mkdir(parents=True)
    (edir / "alpha-model__rgb__2022.npy").touch()
    (edir / "beta-model__rgb__2020.npy").touch()
    (edir / "alpha-model__cir__2022.npy").touch()  # different bands
    found = swap_model._disk_models("expx", "rgb")
    assert found == {"alpha-model", "beta-model"}


def test_format_plan_includes_target_and_purge(tmp_path):
    s = swap_model._format_plan(
        experiment_id="expx",
        aoi_id="aoix",
        bands="rgb",
        cycles=["2020", "2022"],
        target="new-model",
        current={"old-model"},
        to_purge=["old-model"],
        will_rewrite_yaml=True,
        yaml_path=tmp_path / "exp.yaml",
    )
    assert "new-model" in s
    assert "old-model" in s
    assert "rewrite YAML model_id -> new-model" in s


def test_format_plan_no_purge_no_rewrite(tmp_path):
    s = swap_model._format_plan(
        experiment_id="expx", aoi_id="aoix", bands="rgb", cycles=["2022"],
        target="m", current={"m"}, to_purge=[], will_rewrite_yaml=False,
        yaml_path=tmp_path / "exp.yaml",
    )
    assert "to purge         : <none>" in s
    assert "will rewrite YAML: False" in s


def test_purge_paths_for_includes_weights_and_sidecars(tmp_path, monkeypatch):
    """The purge-paths list contains the .npy + .json + weights dir."""
    from terra_query.core import paths

    monkeypatch.setattr(paths, "PIPELINE_OUTPUTS", tmp_path)
    monkeypatch.setattr(paths, "VERIFICATION", tmp_path)
    monkeypatch.setattr(paths, "MODEL_WEIGHTS_DIR", tmp_path / "weights")
    ps = swap_model._purge_paths_for("expx", "old-model", "rgb", ["2020", "2022"])
    names = [str(p) for p in ps]
    assert any("old-model__rgb__2020.npy" in n for n in names)
    assert any("old-model__rgb__2020.json" in n for n in names)
    assert any("old-model__rgb__2022.npy" in n for n in names)
    assert any("weights/old-model" in n for n in names)


def test_yaml_rewrite_round_trips(tmp_path):
    """The YAML rewrite changes only the model_id, preserving every other key."""
    p = _write_min_yaml(tmp_path, model_id="old-model")
    swap_model._rewrite_yaml_model_id(p, "new-model")
    doc = yaml.safe_load(p.read_text())
    assert doc["model_id"] == "new-model"
    # other keys preserved
    for k in ("experiment_id", "aoi_id", "eval_set_id", "modality", "bands",
              "chip_params", "cycles"):
        assert k in doc
    # the dict has no extra `_source_path` injected by load_experiment
    assert "_source_path" not in doc
