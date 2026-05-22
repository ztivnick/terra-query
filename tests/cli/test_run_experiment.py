"""Unit tests for the run_experiment umbrella CLI.

These tests cover the plan generation + stage-selection logic without
actually running any stage subprocess (which would hit the network /
GPU / DB)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from terra_query.cli import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_min_yaml(tmp_path: Path) -> Path:
    base = {
        "experiment_id": "trun",
        "aoi_id": "trun_aoi",
        "eval_set_id": "trun_eval",
        "model_id": "trun_model",
        "modality": "aerial",
        "bands": "rgb",
        "chip_params": {"chip_size_m": 224, "stride_m": 112},
        "cycles": ["2022"],
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(base, sort_keys=False))
    return p


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "terra_query.cli.run_experiment", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# ---- stage table ----

def test_stage_table_is_ten_in_order():
    assert len(run_experiment.STAGES) == 10
    names = [s.name for s in run_experiment.STAGES]
    assert names == [
        "reproject_inputs",
        "discover_aerial",
        "fetch_naip",
        "fetch_sentinel2",
        "build_chip_index",
        "fetch_weights",
        "embed_chips",
        "init_db",
        "load_embeddings",
        "run_n0_retrieval",
    ]


def test_init_db_does_not_take_experiment_arg():
    """init_db is scope-less (DSN-driven); it must NOT receive --experiment."""
    s = run_experiment._stage("init_db")
    assert s.pass_experiment is False
    # every other stage gets --experiment
    for other in run_experiment.STAGES:
        if other.name != "init_db":
            assert other.pass_experiment is True


# ---- range picking ----

def _ns(from_stage=None, to_stage=None, only=None) -> SimpleNamespace:
    return SimpleNamespace(from_stage=from_stage, to_stage=to_stage, only=only)


def test_pick_range_full_pipeline_by_default():
    picked = run_experiment._pick_range(_ns())
    assert len(picked) == len(run_experiment.STAGES)


def test_pick_range_only_returns_single_stage():
    picked = run_experiment._pick_range(_ns(only="build_chip_index"))
    assert len(picked) == 1
    assert picked[0].name == "build_chip_index"


def test_pick_range_from_skips_earlier():
    picked = run_experiment._pick_range(_ns(from_stage="embed_chips"))
    names = [s.name for s in picked]
    assert names == ["embed_chips", "init_db", "load_embeddings", "run_n0_retrieval"]


def test_pick_range_to_stops_after():
    picked = run_experiment._pick_range(_ns(to_stage="build_chip_index"))
    names = [s.name for s in picked]
    assert names == [
        "reproject_inputs", "discover_aerial", "fetch_naip",
        "fetch_sentinel2", "build_chip_index",
    ]


def test_pick_range_from_after_to_errors():
    with pytest.raises(SystemExit, match="is after"):
        run_experiment._pick_range(_ns(from_stage="embed_chips", to_stage="fetch_naip"))


# ---- argv construction ----

def test_build_argv_includes_experiment_for_normal_stage(tmp_path):
    p = tmp_path / "exp.yaml"
    s = run_experiment._stage("fetch_naip")
    argv = run_experiment._build_argv(s, p)
    assert argv[-2:] == ["--experiment", str(p)]


def test_build_argv_omits_experiment_for_init_db(tmp_path):
    p = tmp_path / "exp.yaml"
    s = run_experiment._stage("init_db")
    argv = run_experiment._build_argv(s, p)
    assert "--experiment" not in argv


# ---- end-to-end: subprocess dry-run ----

def test_dry_run_lists_all_ten_stages(tmp_path):
    p = _write_min_yaml(tmp_path)
    r = _run(["--experiment", str(p), "--dry-run"])
    assert r.returncode == 0, r.stderr
    for name in [s.name for s in run_experiment.STAGES]:
        assert name in r.stdout, f"stage {name} missing from dry-run output"
    assert "no stages will run" in r.stdout


def test_dry_run_from_skips_earlier_stages(tmp_path):
    p = _write_min_yaml(tmp_path)
    r = _run(["--experiment", str(p), "--dry-run", "--from", "embed_chips"])
    assert r.returncode == 0
    assert "embed_chips" in r.stdout
    assert "init_db" in r.stdout
    assert "run_n0_retrieval" in r.stdout
    # earlier stages must NOT appear in the planned list (they may still
    # appear as STAGES table entries — the test asserts only that the
    # "stages : N of 10" line says 4 of 10)
    assert "stages          : 4 of 10" in r.stdout


def test_dry_run_only_one_stage(tmp_path):
    p = _write_min_yaml(tmp_path)
    r = _run(["--experiment", str(p), "--dry-run", "--only", "build_chip_index"])
    assert r.returncode == 0
    assert "stages          : 1 of 10" in r.stdout
    assert "build_chip_index" in r.stdout


def test_unknown_stage_errors(tmp_path):
    p = _write_min_yaml(tmp_path)
    r = _run(["--experiment", str(p), "--dry-run", "--only", "totally-fake"])
    assert r.returncode != 0
