"""Tests for the N0 retrieval harness: GT, metrics, random baseline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from terra_query.core import config
from terra_query.core.paths import chip_index_json
from terra_query.eval import n0_retrieval, queries


@pytest.fixture(scope="module")
def cfg() -> dict:
    return config.load_experiment()


@pytest.fixture(scope="module")
def chip_index(cfg):
    p = chip_index_json(config.experiment_id_of(cfg))
    if not p.exists():
        pytest.skip("chip index missing; run `build_chip_index` first")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def eval_features(cfg):
    return n0_retrieval.load_eval_features(config.eval_set_id_of(cfg))


# ---- ground-truth construction ----


def test_chip_location_index_size_matches_grid(chip_index):
    locs = n0_retrieval.chip_location_index(chip_index)
    assert len(locs) == chip_index["n_rows"] * chip_index["n_cols"]


def test_ground_truth_waterfall_includes_bond_falls_chip(chip_index, eval_features):
    """The waterfall concept's GT must include the chip containing Bond Falls."""
    gt = n0_retrieval.build_ground_truth(chip_index, eval_features, findable_only=False)
    locs = n0_retrieval.chip_location_index(chip_index)

    # find bond-falls's eval feature, get its 26916 coord
    bf = next(f for f in eval_features if f["properties"]["id"] == "bond-falls")
    fx, fy = bf["geometry"]["coordinates"]

    # the GT set should be the indices into locs whose bbox contains (fx, fy)
    expected_bf_chips = {
        i for i, ch in enumerate(locs)
        if ch["bbox_26916"][0] <= fx < ch["bbox_26916"][2]
        and ch["bbox_26916"][1] <= fy < ch["bbox_26916"][3]
    }
    assert expected_bf_chips, "no chip contains bond-falls coord"
    assert expected_bf_chips.issubset(gt["waterfall"]), (
        f"waterfall GT missing bond-falls chips: {expected_bf_chips - gt['waterfall']}"
    )


def test_ground_truth_findable_only_excludes_unfindable_features(chip_index, eval_features):
    """Pond concept: strict GT includes the canopy-occluded wetland chip;
    findable-only GT does not."""
    strict = n0_retrieval.build_ground_truth(chip_index, eval_features, findable_only=False)
    findable = n0_retrieval.build_ground_truth(chip_index, eval_features, findable_only=True)
    locs = n0_retrieval.chip_location_index(chip_index)

    wetland = next(f for f in eval_features if f["properties"]["id"] == "unnamed-wetland-w-of-falls")
    assert wetland["properties"]["findable_aerial"] is False
    fx, fy = wetland["geometry"]["coordinates"]
    wetland_chips = {
        i for i, ch in enumerate(locs)
        if ch["bbox_26916"][0] <= fx < ch["bbox_26916"][2]
        and ch["bbox_26916"][1] <= fy < ch["bbox_26916"][3]
    }
    assert wetland_chips.issubset(strict["pond"]), "strict GT must include wetland"
    # findable-only GT does not include the wetland chips IF those chips are not
    # also containers of some other findable pond. With the actual N0 layout
    # the wetland sits in its own chips, so the assertion holds.
    not_findable = wetland_chips - findable["pond"]
    assert not_findable, "findable-only GT unexpectedly includes wetland-only chips"


def test_ground_truth_includes_all_seven_gt_concepts(chip_index, eval_features):
    gt = n0_retrieval.build_ground_truth(chip_index, eval_features, findable_only=False)
    # every GT concept has at least 1 chip (sanity: every N0 feature is in-AOI)
    for c in queries.GT_CONCEPTS:
        assert gt[c], f"concept {c!r} has empty strict GT"


# ---- per-concept metrics (hand-rolled fixture) ----


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_evaluate_concept_known_fixture():
    """10 chips, 2 cycles, 1 GT chip (index 7), known cosines.

    Engineer the data so chip 7 has the highest score, then chip 3, then
    others. Verify hit@1 = 1.0, MRR = 1.0, recall@1 = 1.0.
    """
    rng = np.random.default_rng(13)
    n_chips, d = 10, 8
    # build embeddings such that chip 7 aligns with the query
    text_emb = _normalize(rng.normal(size=(d,)).astype(np.float32))
    cycle_a = rng.normal(size=(n_chips, d)).astype(np.float32)
    cycle_b = rng.normal(size=(n_chips, d)).astype(np.float32)
    # make chip 7 of cycle_a perfectly aligned with text_emb (cos = 1.0)
    cycle_a[7] = text_emb
    cycle_a /= np.linalg.norm(cycle_a, axis=1, keepdims=True)
    cycle_b /= np.linalg.norm(cycle_b, axis=1, keepdims=True)

    res = n0_retrieval.evaluate_concept(
        "waterfall",
        text_emb,
        {"2018": cycle_a, "2022": cycle_b},
        gt_locations={7},
        top_k_max=10,
    )
    assert res.n_gt == 1
    assert res.top_k_indices[0] == 7
    assert res.top_k_cycles[0] == "2018"  # chip 7's max comes from cycle_a
    assert res.hit_at_k[1] == 1.0
    assert res.recall_at_k[1] == 1.0
    assert res.mrr == 1.0


def test_evaluate_concept_empty_gt_returns_zero_metrics():
    rng = np.random.default_rng(17)
    n_chips, d = 5, 4
    text_emb = _normalize(rng.normal(size=(d,)).astype(np.float32))
    arr = rng.normal(size=(n_chips, d)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    res = n0_retrieval.evaluate_concept(
        "abandoned_cabin", text_emb, {"2022": arr}, gt_locations=set(), top_k_max=5,
    )
    assert res.n_gt == 0
    assert res.mrr == 0.0
    for k in res.hit_at_k:
        assert res.hit_at_k[k] == 0.0
        assert res.recall_at_k[k] == 0.0
    # top_k_indices still populated for thumbnail rendering
    assert len(res.top_k_indices) == 5


# ---- random baseline ----


def test_random_hit_at_k_closed_form_matches_empirical():
    n_chips, n_gt = 2209, 3
    for k in [1, 5, 10, 20]:
        closed = n0_retrieval.random_hit_at_k_closed_form(n_chips, n_gt, k)
        emp = n0_retrieval.random_metrics_empirical(n_chips, n_gt, n_trials=2000, seed=42)["hit@k"][k]
        # 2-sigma for a Bernoulli(closed) over 2000 trials
        sigma = max((closed * (1 - closed) / 2000) ** 0.5, 1e-3)
        assert abs(closed - emp) < 3 * sigma, (
            f"K={k}: closed={closed:.4f}, emp={emp:.4f}, sigma={sigma:.4f}"
        )


def test_random_recall_at_k_is_k_over_n():
    assert n0_retrieval.random_recall_at_k(2209, 3, 10) == pytest.approx(10 / 2209)
    assert n0_retrieval.random_recall_at_k(2209, 8, 20) == pytest.approx(20 / 2209)


def test_random_mrr_closed_form_matches_empirical():
    n_chips, n_gt = 100, 5
    closed = n0_retrieval.random_mrr_closed_form(n_chips, n_gt)
    emp = n0_retrieval.random_metrics_empirical(n_chips, n_gt, n_trials=4000, seed=99)["mrr"]
    assert abs(closed - emp) < 0.01, f"closed={closed}, emp={emp}"


# ---- loading combos ----


def test_load_embeddings_for_combo_production_rgb_2022(cfg):
    """Load the production cell as a 1-cycle combo and verify shape +
    chip_id_order is the location-only key (year prefix stripped)."""
    from terra_query.embed import models

    embed_dim = models.spec(models.PRODUCTION_MODEL_ID).embed_dim
    embs, chip_id_order = n0_retrieval.load_embeddings_for_combo(
        config.experiment_id_of(cfg), models.PRODUCTION_MODEL_ID, "rgb", ["2022"]
    )
    assert set(embs) == {"2022"}
    assert embs["2022"].shape[1] == embed_dim
    assert embs["2022"].shape[0] == len(chip_id_order)
    assert chip_id_order[0].startswith("r0") and "_c0" in chip_id_order[0]
    assert not chip_id_order[0].startswith("naip_")
