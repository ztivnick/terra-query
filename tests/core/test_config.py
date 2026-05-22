"""Tests for terra_query.core.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from terra_query.core import config


def _write_minimal(path: Path, **overrides) -> Path:
    """Write a minimal-valid YAML config; overrides replace top-level keys."""
    base = {
        "experiment_id": "test_exp",
        "aoi_id": "test_aoi",
        "eval_set_id": "test_eval",
        "model_id": "test_model",
        "modality": "aerial",
        "bands": "rgb",
        "chip_params": {"chip_size_m": 224, "stride_m": 112},
        "cycles": ["2020", "2022"],
    }
    base.update(overrides)
    import yaml
    path.write_text(yaml.safe_dump(base))
    return path


# ---- resolution order ----

def test_explicit_path_wins(tmp_path, monkeypatch):
    p = _write_minimal(tmp_path / "foo.yaml")
    # set env var to something else to confirm the arg wins
    monkeypatch.setenv(config.CONFIG_ENV_VAR, str(tmp_path / "nonexistent.yaml"))
    cfg = config.load_experiment(p)
    assert config.experiment_id_of(cfg) == "test_exp"


def test_env_var_wins_over_default(tmp_path, monkeypatch):
    p = _write_minimal(tmp_path / "env.yaml", experiment_id="from_env")
    monkeypatch.setenv(config.CONFIG_ENV_VAR, str(p))
    cfg = config.load_experiment()
    assert config.experiment_id_of(cfg) == "from_env"


def test_default_used_when_no_arg_no_env(monkeypatch):
    monkeypatch.delenv(config.CONFIG_ENV_VAR, raising=False)
    cfg = config.load_experiment()
    # default is the bond_falls config; it must exist for the project to work
    assert config.experiment_id_of(cfg) == "bond_falls_25km_poc"


# ---- error paths ----

def test_missing_file_raises_with_helpful_message(tmp_path, monkeypatch):
    monkeypatch.delenv(config.CONFIG_ENV_VAR, raising=False)
    with pytest.raises(SystemExit) as exc_info:
        config.load_experiment(tmp_path / "missing.yaml")
    msg = str(exc_info.value)
    assert "missing.yaml" in msg
    assert "tried:" in msg


def test_non_mapping_yaml_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(SystemExit, match="not a YAML mapping"):
        config.load_experiment(p)


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "incomplete.yaml"
    p.write_text("experiment_id: x\n")
    with pytest.raises(SystemExit) as exc_info:
        config.load_experiment(p)
    msg = str(exc_info.value)
    assert "missing required keys" in msg
    # every required-but-absent key should be named
    for k in config.REQUIRED_KEYS:
        if k != "experiment_id":
            assert k in msg


# ---- accessors ----

def test_accessors_return_typed_values(tmp_path):
    p = _write_minimal(tmp_path / "ok.yaml")
    cfg = config.load_experiment(p)
    assert config.aoi_id_of(cfg) == "test_aoi"
    assert config.eval_set_id_of(cfg) == "test_eval"
    assert config.model_id_of(cfg) == "test_model"
    assert config.modality_of(cfg) == "aerial"
    assert config.bands_of(cfg) == "rgb"
    cp = config.chip_params_of(cfg)
    assert cp == {"chip_size_m": 224, "stride_m": 112}
    assert all(isinstance(v, int) for v in cp.values())
    cycles = config.cycles_of(cfg)
    assert cycles == ["2020", "2022"]
    assert all(isinstance(c, str) for c in cycles)


def test_cycles_int_coerced_to_str(tmp_path):
    p = _write_minimal(tmp_path / "ints.yaml", cycles=[2020, 2022])
    cfg = config.load_experiment(p)
    assert config.cycles_of(cfg) == ["2020", "2022"]


def test_source_path_tracks_load_location(tmp_path):
    p = _write_minimal(tmp_path / "sp.yaml")
    cfg = config.load_experiment(p)
    assert config.source_path_of(cfg) == p


# ---- default-config integration: confirm the real bond_falls YAML loads ----

def test_default_bond_falls_yaml_has_all_required_keys(monkeypatch):
    monkeypatch.delenv(config.CONFIG_ENV_VAR, raising=False)
    cfg = config.load_experiment()
    # spot-check the production values match S05/S06
    assert config.experiment_id_of(cfg) == "bond_falls_25km_poc"
    assert config.aoi_id_of(cfg) == "bond_falls_block"
    assert config.eval_set_id_of(cfg) == "known_features"
    assert config.model_id_of(cfg) == "georsclip-vit-l-14-336"
    assert config.bands_of(cfg) == "rgb"
    assert config.chip_params_of(cfg) == {"chip_size_m": 224, "stride_m": 112}
    assert config.cycles_of(cfg) == ["2012", "2014", "2016", "2018", "2020", "2022"]
