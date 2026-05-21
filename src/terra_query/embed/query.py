"""Query path: prompt ensembling, cycle max-pool, top-K cosine search.

A "chip-location" is one row/col of the chip grid, shared across all
NAIP cycles. The eval harness ranks chip-LOCATIONS (not (chip, cycle)
pairs), so retrieval aggregates per-cycle similarities by max-pool
before ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """Top-K result for one query.

    `chip_indices` are positions into the per-cycle chip array (0..n_chips-1).
    `scores[i]` is the max-pooled cosine for `chip_indices[i]`.
    `cycle_keys[i]` is the cycle key (e.g. "2018") that contributed the max
    for `chip_indices[i]`.
    """

    chip_indices: np.ndarray  # (top_k,) int
    scores: np.ndarray  # (top_k,) float32
    cycle_keys: list[str]  # length top_k


def encode_prompt_ensemble(prompts, model, tokenizer, device) -> np.ndarray:
    """Encode a list of prompts -> single L2-normalized text vector.

    Each prompt is encoded and L2-normalized (encode_text_batch does this),
    then we mean over the N prompts and renormalize. Mean of unit vectors
    is the standard CLIP prompt-ensembling op; the renormalization keeps
    cosine well-defined downstream.
    """
    from terra_query.embed import encoder

    if not prompts:
        raise ValueError("prompts must be non-empty")
    embs = encoder.encode_text_batch(model, tokenizer, prompts, device)
    # embs is (N, D), each row already unit-norm
    mean = embs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise RuntimeError("prompt-ensemble mean is zero; check prompts / model")
    return (mean / norm).astype(np.float32)


def stack_cycle_embeddings(
    embeddings_by_cycle: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Stack per-cycle (n_chips, embed_dim) arrays -> (n_cycles, n_chips, embed_dim).

    Returns the stacked array and the cycle-key order (sorted for stability).
    Validates that every cycle's array shares (n_chips, embed_dim).
    """
    if not embeddings_by_cycle:
        raise ValueError("embeddings_by_cycle must be non-empty")
    cycle_keys = sorted(embeddings_by_cycle.keys())
    first = embeddings_by_cycle[cycle_keys[0]]
    for k in cycle_keys[1:]:
        if embeddings_by_cycle[k].shape != first.shape:
            raise ValueError(
                f"cycle shapes differ: {k} is {embeddings_by_cycle[k].shape}, "
                f"{cycle_keys[0]} is {first.shape}"
            )
    stacked = np.stack([embeddings_by_cycle[k] for k in cycle_keys], axis=0)
    return stacked, cycle_keys


def search(
    text_emb: np.ndarray,
    embeddings_by_cycle: dict[str, np.ndarray],
    top_k: int,
) -> SearchResult:
    """Top-K chip-locations by max-pooled cosine similarity across cycles.

    `text_emb` is a (D,) L2-normalized query vector. Each cycle's
    embeddings are also L2-normalized rows, so cosine = dot.

    Returns SearchResult sorted by descending score.
    """
    if text_emb.ndim != 1:
        raise ValueError(f"text_emb must be 1-D, got {text_emb.shape}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    stacked, cycle_keys = stack_cycle_embeddings(embeddings_by_cycle)
    n_cycles, n_chips, _ = stacked.shape

    # cosine = dot since both sides are unit-normed
    # shape: (n_cycles, n_chips)
    cos_per_cycle = stacked @ text_emb

    # max across the cycle axis per chip-location
    max_cos = cos_per_cycle.max(axis=0)  # (n_chips,)
    argmax_cycle_idx = cos_per_cycle.argmax(axis=0)  # (n_chips,) which cycle won

    # top-K chip indices by max-pooled cosine, descending
    if top_k > n_chips:
        top_k = n_chips
    # partial sort: argpartition then sort the top_k
    part = np.argpartition(-max_cos, top_k - 1)[:top_k]
    order = part[np.argsort(-max_cos[part])]

    return SearchResult(
        chip_indices=order.astype(np.int64),
        scores=max_cos[order].astype(np.float32),
        cycle_keys=[cycle_keys[i] for i in argmax_cycle_idx[order]],
    )


def search_ensemble(
    text_embs_by_model: dict[str, np.ndarray],
    embeddings_by_model_cycle: dict[str, dict[str, np.ndarray]],
    top_k: int,
    primary_model_id: str | None = None,
) -> SearchResult:
    """Top-K chip-locations by ENSEMBLE-AVERAGED max-pooled cosine.

    For each model in the ensemble:
      1. cosine vs query (per cycle), max across cycles per chip-location.
    Average the per-model max-pooled scores per chip-location -> final score.
    Sort, take top_k.

    The returned `cycle_keys` come from the PRIMARY model's argmax (the
    first model in `text_embs_by_model` insertion order, or `primary_model_id`
    if specified). Different models may pick different winning cycles for
    the same chip; the primary's winning cycle is the principled pick for
    downstream thumbnail rendering. The score itself is the cross-model
    average, not the primary alone.
    """
    if not text_embs_by_model:
        raise ValueError("text_embs_by_model must be non-empty")
    if set(text_embs_by_model) != set(embeddings_by_model_cycle):
        raise ValueError(
            f"text_embs models {sorted(text_embs_by_model)} != "
            f"embeddings models {sorted(embeddings_by_model_cycle)}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    model_ids = list(text_embs_by_model.keys())
    if primary_model_id is None:
        primary_model_id = model_ids[0]
    if primary_model_id not in text_embs_by_model:
        raise ValueError(f"primary_model_id {primary_model_id!r} not in ensemble")

    # per-model max-pool across cycles
    per_model_max: dict[str, np.ndarray] = {}
    primary_argmax: np.ndarray | None = None
    primary_cycle_keys: list[str] | None = None
    n_chips: int | None = None
    for mid, text_emb in text_embs_by_model.items():
        if text_emb.ndim != 1:
            raise ValueError(f"text_emb for {mid} must be 1-D, got {text_emb.shape}")
        stacked, cycle_keys = stack_cycle_embeddings(embeddings_by_model_cycle[mid])
        # (n_cycles, n_chips) cosine
        cos_per_cycle = stacked @ text_emb
        max_cos = cos_per_cycle.max(axis=0)
        per_model_max[mid] = max_cos
        if n_chips is None:
            n_chips = max_cos.shape[0]
        elif n_chips != max_cos.shape[0]:
            raise ValueError(
                f"chip-location count differs between models: "
                f"{primary_model_id} has {n_chips}, {mid} has {max_cos.shape[0]}"
            )
        if mid == primary_model_id:
            primary_argmax = cos_per_cycle.argmax(axis=0)
            primary_cycle_keys = cycle_keys

    # mean across models per chip-location
    stacked_per_model = np.stack(list(per_model_max.values()), axis=0)  # (n_models, n_chips)
    ensemble_scores = stacked_per_model.mean(axis=0)  # (n_chips,)

    if top_k > n_chips:
        top_k = n_chips
    part = np.argpartition(-ensemble_scores, top_k - 1)[:top_k]
    order = part[np.argsort(-ensemble_scores[part])]

    return SearchResult(
        chip_indices=order.astype(np.int64),
        scores=ensemble_scores[order].astype(np.float32),
        cycle_keys=[primary_cycle_keys[i] for i in primary_argmax[order]],
    )
