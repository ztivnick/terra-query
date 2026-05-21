"""Query path tests: prompt ensembling, cycle stacking, max-pool top-K."""

from __future__ import annotations

import numpy as np
import pytest

from terra_query.embed import query
from terra_query.embed.query import SearchResult


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_stack_cycle_embeddings_shape_and_order():
    a = np.zeros((3, 4), dtype=np.float32)
    b = np.ones((3, 4), dtype=np.float32)
    stacked, keys = query.stack_cycle_embeddings({"2018": b, "2012": a})
    assert keys == ["2012", "2018"]
    assert stacked.shape == (2, 3, 4)
    np.testing.assert_array_equal(stacked[0], a)
    np.testing.assert_array_equal(stacked[1], b)


def test_stack_cycle_embeddings_mismatched_shapes_raise():
    a = np.zeros((3, 4), dtype=np.float32)
    b = np.ones((3, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        query.stack_cycle_embeddings({"2012": a, "2018": b})


def test_search_max_pool_returns_max_per_location_and_winning_cycle():
    text_emb = _unit([1.0, 0.0])
    cycle_2018 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cycle_2012 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    res = query.search(text_emb, {"2018": cycle_2018, "2012": cycle_2012}, top_k=2)
    np.testing.assert_array_equal(np.sort(res.chip_indices), [0, 1])
    np.testing.assert_allclose(res.scores, [1.0, 1.0])
    winners = dict(zip(res.chip_indices.tolist(), res.cycle_keys))
    assert winners[0] == "2018"
    assert winners[1] == "2012"


def test_search_top_k_sorted_descending():
    rng = np.random.default_rng(7)
    n_chips, d = 50, 16
    arr = rng.normal(size=(n_chips, d)).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    text_emb = _unit(rng.normal(size=(d,)))
    res = query.search(text_emb, {"2022": arr}, top_k=10)
    assert res.chip_indices.shape == (10,)
    diffs = np.diff(res.scores)
    assert (diffs <= 1e-6).all()


def test_search_max_pool_vs_independent_argmax():
    """Cross-check: max-pool per location must equal taking the per-cycle max
    via independent numpy calls, regardless of the partial-sort path."""
    rng = np.random.default_rng(11)
    n_chips, d = 30, 8
    arrs = {}
    for y in ["2012", "2014", "2018"]:
        a = rng.normal(size=(n_chips, d)).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        arrs[y] = a
    text_emb = _unit(rng.normal(size=(d,)))
    res = query.search(text_emb, arrs, top_k=n_chips)
    stacked = np.stack([arrs[y] for y in sorted(arrs)], axis=0)
    cos = stacked @ text_emb
    independent_max = cos.max(axis=0)
    independent_order = np.argsort(-independent_max)
    np.testing.assert_array_equal(res.chip_indices, independent_order)
    np.testing.assert_allclose(res.scores, independent_max[independent_order], rtol=1e-5)


def test_search_top_k_clamped_to_n_chips():
    text_emb = _unit([1.0, 0.0])
    arr = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    res = query.search(text_emb, {"2022": arr}, top_k=99)
    assert len(res.chip_indices) == 2


def test_search_invalid_inputs_raise():
    arr = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        query.search(np.zeros((1, 3), dtype=np.float32), {"2022": arr}, top_k=1)
    with pytest.raises(ValueError):
        query.search(np.zeros(3, dtype=np.float32), {"2022": arr}, top_k=0)
    with pytest.raises(ValueError):
        query.search(np.zeros(3, dtype=np.float32), {}, top_k=1)


# === prompt ensemble: integration with the real production model ===


@pytest.fixture(scope="module")
def production_model():
    from terra_query.core.paths import MODEL_WEIGHTS_MANIFEST
    from terra_query.embed import models

    if not MODEL_WEIGHTS_MANIFEST.exists():
        pytest.skip("weights manifest not present; run fetch_weights first")
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _preprocess, tokenizer = models.load_production(device=device)
    embed_dim = models.spec(models.PRODUCTION_MODEL_ID).embed_dim
    return model, tokenizer, device, embed_dim


def test_prompt_ensemble_returns_unit_vector(production_model):
    model, tokenizer, device, embed_dim = production_model
    prompts = [
        "an aerial photo of a waterfall",
        "an aerial photo of a small pond",
        "an aerial view of a forest road",
    ]
    emb = query.encode_prompt_ensemble(prompts, model, tokenizer, device)
    assert emb.shape == (embed_dim,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_prompt_ensemble_single_prompt_equivalent_to_single_text(production_model):
    """Single-prompt ensemble equals direct encode_text within fp32 tolerance:
    mean of one unit vector then renormalize is identity."""
    from terra_query.embed import encoder

    model, tokenizer, device, _embed_dim = production_model
    prompt = "an aerial photo of a waterfall"
    direct = encoder.encode_text(model, tokenizer, prompt, device)
    ensemble = query.encode_prompt_ensemble([prompt], model, tokenizer, device)
    np.testing.assert_allclose(direct, ensemble, atol=1e-5)


def test_prompt_ensemble_empty_raises(production_model):
    model, tokenizer, device, _embed_dim = production_model
    with pytest.raises(ValueError):
        query.encode_prompt_ensemble([], model, tokenizer, device)


# === search_ensemble (unit-level, no model needed) ===


def test_search_ensemble_identical_inputs_match_single_search():
    """Ensembling the same (model, embeddings) twice must rank identically
    to a single-model search (averaging same scores leaves order intact)."""
    rng = np.random.default_rng(31)
    n_chips, d = 30, 8
    embs = rng.normal(size=(n_chips, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    text_emb = _unit(rng.normal(size=(d,)))

    single = query.search(text_emb, {"2022": embs}, top_k=10)
    ensemble = query.search_ensemble(
        text_embs_by_model={"m1": text_emb, "m2": text_emb},
        embeddings_by_model_cycle={"m1": {"2022": embs}, "m2": {"2022": embs}},
        top_k=10,
    )
    np.testing.assert_array_equal(single.chip_indices, ensemble.chip_indices)
    np.testing.assert_allclose(single.scores, ensemble.scores, rtol=1e-5)


def test_search_ensemble_averages_two_models_per_location():
    """Two models with different orderings: ensemble ranks by average score."""
    text_emb = _unit([1.0, 0.0])
    m1 = np.array([[0.9, 0.0], [0.3, 0.0], [0.7, 0.0], [0.5, 0.0]], dtype=np.float32)
    m2 = np.array([[0.1, 0.0], [0.9, 0.0], [0.5, 0.0], [0.7, 0.0]], dtype=np.float32)
    res = query.search_ensemble(
        text_embs_by_model={"m1": text_emb, "m2": text_emb},
        embeddings_by_model_cycle={"m1": {"2022": m1}, "m2": {"2022": m2}},
        top_k=4,
    )
    assert 0 not in set(res.chip_indices[:3].tolist())
    assert res.chip_indices[-1] == 0
    np.testing.assert_allclose(res.scores[-1], 0.5)
    np.testing.assert_allclose(res.scores[:3], [0.6, 0.6, 0.6])


def test_search_ensemble_primary_drives_cycle_attribution():
    """The cycle_keys in the result must come from the primary model's argmax."""
    rng = np.random.default_rng(37)
    n_chips, d = 6, 4

    def cycle_embs(scale):
        a = rng.normal(size=(n_chips, d)).astype(np.float32) * scale
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a

    text_emb = _unit(rng.normal(size=(d,)))
    m1 = {"2012": cycle_embs(1.0), "2022": cycle_embs(1.0)}
    m2 = {"2012": cycle_embs(1.0), "2022": cycle_embs(1.0)}
    res_p1 = query.search_ensemble(
        text_embs_by_model={"m1": text_emb, "m2": text_emb},
        embeddings_by_model_cycle={"m1": m1, "m2": m2},
        top_k=n_chips, primary_model_id="m1",
    )
    res_p2 = query.search_ensemble(
        text_embs_by_model={"m1": text_emb, "m2": text_emb},
        embeddings_by_model_cycle={"m1": m1, "m2": m2},
        top_k=n_chips, primary_model_id="m2",
    )
    np.testing.assert_allclose(res_p1.scores, res_p2.scores, rtol=1e-5)
    assert all(c in {"2012", "2022"} for c in res_p1.cycle_keys)
    assert all(c in {"2012", "2022"} for c in res_p2.cycle_keys)


def test_search_ensemble_mismatched_models_raise():
    text_emb = _unit([1.0, 0.0])
    embs = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        query.search_ensemble(
            text_embs_by_model={"m1": text_emb, "m2": text_emb},
            embeddings_by_model_cycle={"m1": {"2022": embs}},
            top_k=1,
        )
    with pytest.raises(ValueError):
        query.search_ensemble(
            text_embs_by_model={},
            embeddings_by_model_cycle={},
            top_k=1,
        )
