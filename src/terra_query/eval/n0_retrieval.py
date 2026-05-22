"""Retrieval harness: load embeddings, build GT from the eval set,
score, report.

Produces hit@K, recall@K, MRR per concept per (model, bands), plus a
closed-form + sampled random baseline.

Vocabulary:
- "chip-location": one position in the per-cycle chip array. The set
  is shared across every NAIP cycle (the chip grid is defined in
  EPSG:26916 and applied per cycle).
- "GT chip-location for concept C": any chip-location whose bbox in
  26916 contains the 26916 point coord of some eval feature whose
  `type` maps to concept C.

From S6 onward, the production search path goes through the pgvector
store (`evaluate_concept_via_db`). The in-memory `evaluate_concept`
stays as a pure-math unit-test fixture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import psycopg

from terra_query.core.paths import (
    chip_index_json,
    embeddings_json,
    embeddings_npy,
    eval_26916,
)
from terra_query.embed.models import PRODUCTION_MODEL_ID
from terra_query.embed.query import SearchResult, search
from terra_query.eval.queries import CONCEPT_TO_N0_TYPE
from terra_query.vector_store.search import search as vs_search

# K values we report for hit@K / recall@K. Capped at 20.
K_VALUES = (1, 5, 10, 20)


def load_chip_index(experiment_id: str) -> dict:
    return json.loads(chip_index_json(experiment_id).read_text())


def load_eval_features(eval_set_id: str) -> list[dict]:
    return json.loads(eval_26916(eval_set_id).read_text())["features"]


def _cycle_block(chip_index: dict, year: str) -> dict:
    for c in chip_index["cycles"]:
        if c["year"] == year:
            return c
    raise KeyError(f"cycle {year} not in chip index")


def chip_location_index(chip_index: dict) -> list[dict]:
    """The 2,209 chip-locations, in the same order embedded files use.

    Same as `cycles[year=2022].chips` (any cycle works; the per-cycle
    chip order is identical by construction). Each chip is a dict with
    `chip_id`, `row`, `col`, `bbox_26916`, `center_26916`, `center_wgs84`,
    `inside_aoi`.
    """
    return _cycle_block(chip_index, chip_index["cycles"][0]["year"])["chips"]


def chip_location_to_index_map(chip_index: dict) -> dict[str, int]:
    """Map chip_location_id ("r034_c028") -> 0-based row index in chip_location_index.

    Cycle-agnostic (uses row/col directly). The vector store returns
    chip_location_id strings; this is how we map them back to the integer
    indices used by GT sets.
    """
    locations = chip_location_index(chip_index)
    return {f"r{loc['row']:03d}_c{loc['col']:03d}": i for i, loc in enumerate(locations)}


def build_ground_truth(
    chip_index: dict,
    eval_features: list[dict],
    findable_only: bool,
) -> dict[str, set[int]]:
    """Return concept -> set of chip-location indices.

    A chip-location is in GT for concept C if its bbox in 26916 contains
    the 26916 point coord of some N0 feature whose `type` maps to C
    (via `CONCEPT_TO_N0_TYPE`). Under 50% chip overlap, that's typically
    1-4 chips per feature.

    If `findable_only` is True, features with `findable_aerial == False`
    are skipped (the wetland under canopy, the cemetery under tree cover).
    """
    locations = chip_location_index(chip_index)
    # invert concept -> n0_type to n0_type -> concept
    type_to_concept: dict[str, str] = {}
    for concept, n0_type in CONCEPT_TO_N0_TYPE.items():
        if n0_type is not None:
            type_to_concept[n0_type] = concept

    gt: dict[str, set[int]] = {c: set() for c in CONCEPT_TO_N0_TYPE if CONCEPT_TO_N0_TYPE[c] is not None}

    for feat in eval_features:
        n0_type = feat["properties"]["type"]
        if n0_type not in type_to_concept:
            continue  # this feature's type doesn't correspond to any concept (shouldn't happen given queries.py was built from N0)
        concept = type_to_concept[n0_type]
        if findable_only and not feat["properties"].get("findable_aerial", False):
            continue
        fx, fy = feat["geometry"]["coordinates"]
        for idx, ch in enumerate(locations):
            w, s, e, n = ch["bbox_26916"]
            if w <= fx < e and s <= fy < n:
                gt[concept].add(idx)
    return gt


# ---- per-concept metrics ----

@dataclass(frozen=True)
class ConceptResult:
    concept: str
    n_gt: int
    hit_at_k: dict[int, float]   # K -> 0/1
    recall_at_k: dict[int, float]  # K -> [0,1]
    mrr: float
    top_k_indices: list[int]
    top_k_scores: list[float]
    top_k_cycles: list[str]


def _compute_concept_metrics(
    concept: str,
    top_indices: list[int],
    top_scores: list[float],
    top_cycles: list[str],
    gt_locations: set[int],
) -> ConceptResult:
    """Pure metric computation: hit@K, recall@K, MRR over a pre-ranked list."""
    hit_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    for k in K_VALUES:
        if k > len(top_indices):
            break
        top_set = set(top_indices[:k])
        inter = top_set & gt_locations
        hit_at_k[k] = 1.0 if inter else 0.0
        recall_at_k[k] = (len(inter) / len(gt_locations)) if gt_locations else 0.0

    mrr = 0.0
    for rank, idx in enumerate(top_indices, start=1):
        if idx in gt_locations:
            mrr = 1.0 / rank
            break

    return ConceptResult(
        concept=concept,
        n_gt=len(gt_locations),
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        top_k_indices=top_indices,
        top_k_scores=top_scores,
        top_k_cycles=top_cycles,
    )


def evaluate_concept(
    concept: str,
    ensemble_emb: np.ndarray,
    embeddings_by_cycle: dict[str, np.ndarray],
    gt_locations: set[int],
    top_k_max: int = max(K_VALUES),
) -> ConceptResult:
    """In-memory path: numpy search + metrics. Used by tests + ensemble CLI.

    The production gate path is `evaluate_concept_via_db`. This function
    is kept as the pure-math fixture so the metric logic is testable
    without a database.
    """
    res: SearchResult = search(ensemble_emb, embeddings_by_cycle, top_k=top_k_max)
    return _compute_concept_metrics(
        concept,
        top_indices=res.chip_indices.tolist(),
        top_scores=res.scores.tolist(),
        top_cycles=list(res.cycle_keys),
        gt_locations=gt_locations,
    )


def evaluate_concept_via_db(
    concept: str,
    ensemble_emb: np.ndarray,
    chip_location_to_index: dict[str, int],
    gt_locations: set[int],
    top_k_max: int = max(K_VALUES),
    model_id: str = PRODUCTION_MODEL_ID,
    bands: str = "rgb",
    conn: psycopg.Connection | None = None,
    overfetch: int | None = None,
) -> ConceptResult:
    """DB-backed path: pgvector HNSW + max-pool in SQL + metrics.

    `chip_location_to_index` maps the DB's chip_location_id strings back
    to the 0-based row index used by GT sets. Build it once per run via
    `chip_location_to_index_map(chip_index)`.
    """
    hits = vs_search(
        ensemble_emb,
        top_k=top_k_max,
        model_id=model_id,
        bands=bands,
        conn=conn,
        overfetch=overfetch,
    )
    top_indices = [chip_location_to_index[h.chip_location_id] for h in hits]
    top_scores = [h.score for h in hits]
    top_cycles = [h.winning_cycle for h in hits]
    return _compute_concept_metrics(
        concept, top_indices, top_scores, top_cycles, gt_locations,
    )


# ---- random baseline ----

def random_hit_at_k_closed_form(n_chips: int, n_gt: int, k: int) -> float:
    """P(at least one of n_gt GT chips is in a uniformly-random top-K).

    Closed form: 1 - C(N-g, K)/C(N, K) = 1 - prod_i (N-g-i)/(N-i).
    """
    if n_gt <= 0 or k <= 0 or n_chips <= 0:
        return 0.0
    if k >= n_chips:
        return 1.0
    miss = 1.0
    for i in range(k):
        num = n_chips - n_gt - i
        den = n_chips - i
        if num <= 0:
            return 1.0  # impossible to miss; some GT must land in top-K
        miss *= num / den
    return 1.0 - miss


def random_recall_at_k(n_chips: int, n_gt: int, k: int) -> float:
    """Expected fraction of GT in a random top-K = K/N (hypergeometric mean)."""
    if n_gt <= 0 or n_chips <= 0:
        return 0.0
    return min(k, n_chips) / n_chips


def random_mrr_closed_form(n_chips: int, n_gt: int) -> float:
    """E[1/rank of first GT] for uniform random ranking.

    For g GT among N, P(first GT at rank k) = C(N-k, g-1) / C(N, g) for
    k in [1, N-g+1]. E[1/rank] is sum_k (1/k) * P(first GT at rank k).
    """
    if n_gt <= 0 or n_chips <= 0:
        return 0.0
    total = 0.0
    denom = math.comb(n_chips, n_gt)
    for k in range(1, n_chips - n_gt + 2):
        num = math.comb(n_chips - k, n_gt - 1)
        total += (num / denom) / k
    return total


def random_metrics_empirical(
    n_chips: int, n_gt: int, n_trials: int = 1000, seed: int = 0
) -> dict:
    """Sampled random baseline as a sanity check on the closed-form values."""
    rng = np.random.default_rng(seed)
    hit = {k: 0.0 for k in K_VALUES}
    recall = {k: 0.0 for k in K_VALUES}
    mrr_sum = 0.0
    for _ in range(n_trials):
        perm = rng.permutation(n_chips)
        gt_set = set(rng.choice(n_chips, size=n_gt, replace=False).tolist())
        for k in K_VALUES:
            if k > n_chips:
                continue
            top_set = set(perm[:k].tolist())
            inter = top_set & gt_set
            if inter:
                hit[k] += 1.0
            recall[k] += len(inter) / max(n_gt, 1)
        for rank, idx in enumerate(perm.tolist(), start=1):
            if idx in gt_set:
                mrr_sum += 1.0 / rank
                break
    return {
        "hit@k": {k: hit[k] / n_trials for k in K_VALUES if k <= n_chips},
        "recall@k": {k: recall[k] / n_trials for k in K_VALUES if k <= n_chips},
        "mrr": mrr_sum / n_trials,
    }


# ---- loading the per-cycle embedding artifacts ----

def load_embeddings_for_combo(
    experiment_id: str,
    model_id: str,
    bands: str,
    cycles: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load per-cycle .npy + sidecar for one (model, bands) combo.

    Returns (cycle -> (n_chips, embed_dim) arr, chip_id_order).
    Validates that every cycle's sidecar agrees on the chip_id order.
    """
    embeddings: dict[str, np.ndarray] = {}
    chip_id_order: list[str] | None = None
    for cycle in cycles:
        npy = embeddings_npy(experiment_id, model_id, bands, cycle)
        js = embeddings_json(experiment_id, model_id, bands, cycle)
        if not (npy.exists() and js.exists()):
            raise FileNotFoundError(f"missing embedding for ({model_id}, {bands}, {cycle})")
        arr = np.load(npy)
        sc = json.loads(js.read_text())
        # row/col part of each chip_id is the chip-location id; strip year
        stripped = [
            cid.replace(f"naip_{cycle}_", "", 1) for cid in sc["chip_ids"]
        ]
        if chip_id_order is None:
            chip_id_order = stripped
        elif stripped != chip_id_order:
            raise RuntimeError(
                f"chip_id order mismatch between cycles for {model_id}/{bands}: "
                f"{cycle} vs {cycles[0]}"
            )
        embeddings[cycle] = arr
    if chip_id_order is None:
        raise ValueError("no cycles provided")
    return embeddings, chip_id_order
