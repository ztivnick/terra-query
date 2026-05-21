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
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from terra_query.core.paths import (
    CHIP_INDEX_JSON,
    EVAL_26916,
    embeddings_json,
    embeddings_npy,
)
from terra_query.embed.query import SearchResult, search
from terra_query.eval.queries import CONCEPT_TO_N0_TYPE

# K values we report for hit@K / recall@K. Capped at 20.
K_VALUES = (1, 5, 10, 20)


def load_chip_index() -> dict:
    return json.loads(CHIP_INDEX_JSON.read_text())


def load_eval_features() -> list[dict]:
    return json.loads(EVAL_26916.read_text())["features"]


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


def evaluate_concept(
    concept: str,
    ensemble_emb: np.ndarray,
    embeddings_by_cycle: dict[str, np.ndarray],
    gt_locations: set[int],
    top_k_max: int = max(K_VALUES),
) -> ConceptResult:
    """Compute hit@K, recall@K, MRR for one concept's prompt-ensembled query.

    For no-GT concepts, pass `gt_locations = set()`: metrics will be
    0/0 but `top_k_indices` will still rank the chips so thumbnails
    can be rendered for qualitative inspection.
    """
    res: SearchResult = search(ensemble_emb, embeddings_by_cycle, top_k=top_k_max)
    top_indices = res.chip_indices.tolist()
    top_scores = res.scores.tolist()
    top_cycles = list(res.cycle_keys)

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
        npy = embeddings_npy(model_id, bands, cycle)
        js = embeddings_json(model_id, bands, cycle)
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
