"""Tests for the frozen concept -> prompt-ensemble mapping."""

from __future__ import annotations

from terra_query.eval import queries


def test_gt_and_no_gt_concept_counts_match_eval_set():
    """8 GT concepts (1 per eval-set type: waterfall, pond, dam, road,
    parking, boardwalk, cemetery, cabin) + 4 no-GT concepts for
    qualitative top-K."""
    assert len(queries.GT_CONCEPTS) == 8
    assert len(queries.NO_GT_CONCEPTS) == 4
    assert "abandoned_cabin" in queries.GT_CONCEPTS
    assert "abandoned_cabin" not in queries.NO_GT_CONCEPTS


def test_each_concept_has_at_least_6_prompts():
    for c, prompts in queries.all_concepts().items():
        assert len(prompts) >= 6, f"{c} has only {len(prompts)} prompts"


def test_concept_to_n0_type_covers_every_concept():
    assert set(queries.CONCEPT_TO_N0_TYPE) == set(queries.all_concepts())


def test_gt_concept_n0_types_match_known_eval_set():
    """Every GT concept's eval-type must appear in the eval set, and vice versa."""
    import json

    from terra_query.core.paths import EVAL_26916

    eval_types = {
        f["properties"]["type"] for f in json.loads(EVAL_26916.read_text())["features"]
    }
    gt_types = {queries.CONCEPT_TO_N0_TYPE[c] for c in queries.GT_CONCEPTS}
    assert gt_types == eval_types, f"GT concept types {gt_types} != eval set types {eval_types}"


def test_no_gt_concepts_have_none_n0_type():
    for c in queries.NO_GT_CONCEPTS:
        assert queries.CONCEPT_TO_N0_TYPE[c] is None
