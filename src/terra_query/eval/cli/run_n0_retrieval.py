"""CLI: run the retrieval gate harness across the configured (model, bands) cells.

Loads chip index + eval features, builds GT (findable + strict), loops over
every (model, bands) combo: encode every concept's prompt ensemble, search
across the per-cycle embeddings with max-pool, compute hit/recall/MRR,
render top-10 thumbnails per concept. Writes (paths resolved per
`core/paths.py` from the experiment id):
  - <gate_dir>/n0_retrieval_results.json
  - <gate_dir>/n0_retrieval_report.md
  - <gate_dir>/topk_chips/<concept>__<model>__<bands>/rank_<R>_<chip>.png
Then prints the automated verdict (5 PASS/FAIL checks).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from terra_query.core import config
from terra_query.core.paths import (
    MODEL_WEIGHTS_MANIFEST,
    n0_report_md,
    n0_results_json,
    naip_cog,
    topk_chip_dir,
    topk_chips_dir,
)
from terra_query.embed import models, query
from terra_query.eval import n0_retrieval, queries
from terra_query.eval.n0_retrieval import K_VALUES
from terra_query.ingest.chips import RGB_BANDS, ChipBox, read_chip
from terra_query.vector_store.db import connect as vs_connect

ALL_BANDS = ["rgb", "cir"]
TOPK_THUMBNAILS = 10


def _config_key(model_id: str, bands: str) -> str:
    return f"{model_id}__{bands}"


def _chip_box(loc: dict) -> ChipBox:
    return ChipBox(row=loc["row"], col=loc["col"],
                   west=loc["bbox_26916"][0], south=loc["bbox_26916"][1])


def render_thumbnails(
    experiment_id: str,
    aoi_id: str,
    fallback_cycle: str,
    concept: str,
    model_id: str,
    bands: str,
    top_k_indices: list[int],
    top_k_cycles: list[str],
    top_k_scores: list[float],
    locations: list[dict],
    force: bool = False,
) -> Path:
    """Render top-K chip PNGs to topk_chips/<concept>__<model>__<bands>/.

    Uses each chip's WINNING cycle (the cycle whose embedding gave the
    max-pool score) so the thumbnail shows the model actually saw that
    appearance. Falls back to `fallback_cycle` if a winning cycle's COG
    is missing.
    """
    from PIL import Image

    out_dir = topk_chip_dir(experiment_id, concept, model_id, bands)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (idx, cycle, score) in enumerate(
        zip(top_k_indices[:TOPK_THUMBNAILS], top_k_cycles[:TOPK_THUMBNAILS], top_k_scores[:TOPK_THUMBNAILS]),
        start=1,
    ):
        loc = locations[idx]
        chip_id_location = f"r{loc['row']:03d}_c{loc['col']:03d}"
        out_path = out_dir / f"rank_{rank:02d}__{chip_id_location}__cyc{cycle}__sim{score:.3f}.png"
        if out_path.exists() and not force:
            continue
        cog = naip_cog(aoi_id, cycle)
        if not cog.exists():
            cog = naip_cog(aoi_id, fallback_cycle)
        arr = read_chip(_chip_box(loc), cog, bands=RGB_BANDS)  # (3, h, w) uint8
        hwc = np.transpose(arr, (1, 2, 0))
        Image.fromarray(hwc, mode="RGB").save(out_path)
    return out_dir


def evaluate_one_combo(
    experiment_id: str,
    aoi_id: str,
    fallback_cycle: str,
    model_id: str,
    bands: str,
    cycles: list[str],
    concepts: dict[str, list[str]],
    gt_findable: dict[str, set[int]],
    gt_strict: dict[str, set[int]],
    locations: list[dict],
    device: str,
    weights_entry: dict,
    chip_loc_to_idx: dict[str, int],
    db_conn,
) -> dict:
    """Encode every concept's prompts and run search via the vector store."""
    t_load = time.time()
    print(f"\n=== {model_id} / {bands} ===")
    print(f"[{model_id}/{bands}] loading model for text encoding ...")
    model, _preprocess, tokenizer = models.load(
        model_id, weights_path=weights_entry.get("weights_path"), device=device
    )
    print(f"[{model_id}/{bands}] model loaded in {time.time() - t_load:.1f}s")

    # pre-flight: rows must exist in the DB for this (model, bands)
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM chip_embeddings "
            "WHERE model_id = %s AND bands = %s",
            (model_id, bands),
        )
        n_rows = cur.fetchone()[0]
    if n_rows == 0:
        raise SystemExit(
            f"chip_embeddings has 0 rows for ({model_id}, {bands}). "
            f"Run `python -m terra_query.vector_store.cli.load_embeddings "
            f"--model {model_id} --bands {bands}` first."
        )
    print(f"[{model_id}/{bands}] {n_rows} rows in vector store")

    combo_out: dict[str, dict] = {}
    try:
        for concept, prompts in concepts.items():
            t0 = time.time()
            text_emb = query.encode_prompt_ensemble(prompts, model, tokenizer, device)
            gt_f = gt_findable.get(concept, set())
            gt_s = gt_strict.get(concept, set())

            res_findable = n0_retrieval.evaluate_concept_via_db(
                concept, text_emb, chip_loc_to_idx, gt_f,
                top_k_max=max(K_VALUES),
                model_id=model_id, bands=bands, conn=db_conn,
            )
            res_strict = n0_retrieval.evaluate_concept_via_db(
                concept, text_emb, chip_loc_to_idx, gt_s,
                top_k_max=max(K_VALUES),
                model_id=model_id, bands=bands, conn=db_conn,
            )

            render_thumbnails(
                experiment_id, aoi_id, fallback_cycle,
                concept, model_id, bands,
                res_findable.top_k_indices, res_findable.top_k_cycles,
                res_findable.top_k_scores, locations,
            )

            combo_out[concept] = {
                "findable": asdict(res_findable),
                "strict": asdict(res_strict),
                "n_prompts": len(prompts),
                "elapsed_s": round(time.time() - t0, 2),
            }
            print(
                f"  [{concept:20s}] n_gt_findable={res_findable.n_gt:2d} "
                f"hit@10={res_findable.hit_at_k.get(10, 0):.0f} "
                f"mrr={res_findable.mrr:.3f} "
                f"top1=chip{res_findable.top_k_indices[0]} score={res_findable.top_k_scores[0]:.3f}"
            )
    finally:
        del model
        import torch

        if device == "mps":
            torch.mps.empty_cache()
    return combo_out


def evaluate_ensemble(
    experiment_id: str,
    aoi_id: str,
    fallback_cycle: str,
    member_model_ids: list[str],
    bands: str,
    cycles: list[str],
    concepts: dict[str, list[str]],
    gt_findable: dict[str, set[int]],
    gt_strict: dict[str, set[int]],
    locations: list[dict],
    device: str,
    weights_manifest: dict,
    primary_model_id: str | None = None,
) -> dict:
    """Encode prompts per ensemble member, search via cross-model averaging.

    `member_model_ids[0]` is the primary by default (drives top-K thumbnail
    cycle reporting). All members must share `bands` (we're ensembling models,
    not band-combos).
    """
    if not member_model_ids:
        raise ValueError("ensemble must have at least one member")
    if primary_model_id is None:
        primary_model_id = member_model_ids[0]
    print(f"\n=== ensemble: {'+'.join(member_model_ids)} / {bands} ===")

    # load each member's text encoder + embeddings
    text_models: dict[str, tuple] = {}  # model_id -> (model, tokenizer)
    embeddings_by_member: dict[str, dict[str, np.ndarray]] = {}
    t_load = time.time()
    for mid in member_model_ids:
        weights_entry = weights_manifest["models"][mid]
        print(f"[ensemble/{mid}] loading text encoder ...")
        model, _preprocess, tokenizer = models.load(
            mid, weights_path=weights_entry.get("weights_path"), device=device
        )
        text_models[mid] = (model, tokenizer)
        embs, _ = n0_retrieval.load_embeddings_for_combo(experiment_id, mid, bands, cycles)
        embeddings_by_member[mid] = embs
    print(f"[ensemble] all {len(member_model_ids)} members loaded in {time.time() - t_load:.1f}s")

    cfg_key = _ensemble_key(member_model_ids, bands)
    combo_out: dict[str, dict] = {}
    try:
        for concept, prompts in concepts.items():
            t0 = time.time()
            # encode prompt ensemble PER member (each model has its own text encoder
            # and vector space; ensembling vectors across models is only valid AFTER
            # cosine, not before).
            text_embs_by_model = {
                mid: query.encode_prompt_ensemble(prompts, mdl, tok, device)
                for mid, (mdl, tok) in text_models.items()
            }
            gt_f = gt_findable.get(concept, set())
            gt_s = gt_strict.get(concept, set())

            # search via cross-model averaging
            res_f_sr = query.search_ensemble(
                text_embs_by_model, embeddings_by_member,
                top_k=max(K_VALUES), primary_model_id=primary_model_id,
            )
            res_f = _wrap_search_result_into_concept_result(concept, res_f_sr, gt_f)
            res_s_sr = query.search_ensemble(
                text_embs_by_model, embeddings_by_member,
                top_k=max(K_VALUES), primary_model_id=primary_model_id,
            )
            res_s = _wrap_search_result_into_concept_result(concept, res_s_sr, gt_s)

            # thumbnails: use the ensemble config key as the folder prefix
            # so they don't collide with single-model thumbnails
            render_thumbnails(
                experiment_id, aoi_id, fallback_cycle,
                concept, cfg_key, bands,
                res_f.top_k_indices, res_f.top_k_cycles, res_f.top_k_scores, locations,
            )
            combo_out[concept] = {
                "findable": asdict(res_f),
                "strict": asdict(res_s),
                "n_prompts": len(prompts),
                "elapsed_s": round(time.time() - t0, 2),
            }
            print(
                f"  [{concept:20s}] n_gt_findable={res_f.n_gt:2d} "
                f"hit@10={res_f.hit_at_k.get(10, 0):.0f} "
                f"mrr={res_f.mrr:.3f} "
                f"top1=chip{res_f.top_k_indices[0]} score={res_f.top_k_scores[0]:.3f}"
            )
    finally:
        for mid, (mdl, _) in text_models.items():
            del mdl
        import torch

        if device == "mps":
            torch.mps.empty_cache()
    return combo_out


def _ensemble_key(member_model_ids: list[str], bands: str) -> str:
    return f"ensemble:{'+'.join(member_model_ids)}__{bands}"


def _wrap_search_result_into_concept_result(
    concept: str,
    sr: query.SearchResult,
    gt_locations: set[int],
):
    """Build a ConceptResult from a SearchResult + GT.

    Mirrors n0_retrieval.evaluate_concept but takes a pre-computed search
    result so we can use the ensemble search path without re-running cosine.
    """
    top_indices = sr.chip_indices.tolist()
    top_scores = sr.scores.tolist()
    top_cycles = list(sr.cycle_keys)
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
    return n0_retrieval.ConceptResult(
        concept=concept,
        n_gt=len(gt_locations),
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        mrr=mrr,
        top_k_indices=top_indices,
        top_k_scores=top_scores,
        top_k_cycles=top_cycles,
    )


def build_random_baseline(n_chips: int, gt_findable: dict, gt_strict: dict) -> dict:
    """Per-concept random hit@K + MRR (closed-form) + an aggregate sample."""
    out: dict = {"per_concept": {}, "aggregate": {}}
    for concept in queries.GT_CONCEPTS:
        nf = len(gt_findable.get(concept, set()))
        ns = len(gt_strict.get(concept, set()))
        out["per_concept"][concept] = {
            "findable": {
                "n_gt": nf,
                "hit@k": {
                    k: n0_retrieval.random_hit_at_k_closed_form(n_chips, nf, k)
                    for k in K_VALUES
                },
                "recall@k": {
                    k: n0_retrieval.random_recall_at_k(n_chips, nf, k)
                    for k in K_VALUES
                },
                "mrr": n0_retrieval.random_mrr_closed_form(n_chips, nf) if nf > 0 else 0.0,
            },
            "strict": {
                "n_gt": ns,
                "hit@k": {
                    k: n0_retrieval.random_hit_at_k_closed_form(n_chips, ns, k)
                    for k in K_VALUES
                },
                "recall@k": {
                    k: n0_retrieval.random_recall_at_k(n_chips, ns, k)
                    for k in K_VALUES
                },
                "mrr": n0_retrieval.random_mrr_closed_form(n_chips, ns) if ns > 0 else 0.0,
            },
        }
    # mean across GT concepts
    out["aggregate"]["findable"] = {
        "hit@k": {
            k: float(np.mean([
                out["per_concept"][c]["findable"]["hit@k"][k] for c in queries.GT_CONCEPTS
            ]))
            for k in K_VALUES
        },
        "mrr": float(np.mean([
            out["per_concept"][c]["findable"]["mrr"] for c in queries.GT_CONCEPTS
        ])),
    }
    out["aggregate"]["strict"] = {
        "hit@k": {
            k: float(np.mean([
                out["per_concept"][c]["strict"]["hit@k"][k] for c in queries.GT_CONCEPTS
            ]))
            for k in K_VALUES
        },
        "mrr": float(np.mean([
            out["per_concept"][c]["strict"]["mrr"] for c in queries.GT_CONCEPTS
        ])),
    }
    return out


# ---- markdown report ----


def _fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def _fmt_score(x: float) -> str:
    return f"{x:.3f}"


def _findable_gt_concepts(per_config_results: dict) -> list[str]:
    """GT concepts where at least one config reports n_gt_findable > 0.

    Aggregate findable metrics should be over concepts where finding is
    even possible. Concepts that are `findable_aerial = false` for every
    feature in the eval set (cemetery, cabin) would always contribute 0
    to findable metrics and unfairly drag down aerial-only aggregates.
    """
    findable = []
    for c in queries.GT_CONCEPTS:
        for cres in per_config_results.values():
            if c in cres and cres[c]["findable"]["n_gt"] > 0:
                findable.append(c)
                break
    return findable


def _aggregate_metric(
    per_config_results: dict, gt_variant: str, metric: str, k: int | None = None,
    concepts: list[str] | None = None,
) -> dict[str, float]:
    """For each config, compute mean over `concepts` of the named metric.

    Default `concepts` = all GT_CONCEPTS. For findable aggregates the caller
    should pass `_findable_gt_concepts(per_config_results)` so unfindable
    concepts (cemetery, cabin under aerial-only) don't drag the mean.
    """
    if concepts is None:
        concepts = list(queries.GT_CONCEPTS)
    out: dict[str, float] = {}
    for cfg, cres in per_config_results.items():
        vals = []
        for c in concepts:
            if c not in cres:
                continue
            r = cres[c][gt_variant]
            if metric == "mrr":
                vals.append(r["mrr"])
            elif metric == "hit@k":
                vals.append(r["hit_at_k"].get(str(k), r["hit_at_k"].get(k, 0.0)))
            elif metric == "recall@k":
                vals.append(r["recall_at_k"].get(str(k), r["recall_at_k"].get(k, 0.0)))
        out[cfg] = float(np.mean(vals)) if vals else 0.0
    return out


def _config_label(cfg_key: str) -> str:
    return cfg_key.replace("__", " / ")


def write_markdown_report(
    experiment_id: str,
    results: dict,
    gt_findable: dict[str, set[int]],
    gt_strict: dict[str, set[int]],
    out_path: Path | None = None,
) -> None:
    """Write the human-readable gate report."""
    if out_path is None:
        out_path = n0_report_md(experiment_id)
    per_config = results["per_config"]
    random_bl = results["random_baseline"]
    n_chips = results["n_chips"]

    lines: list[str] = []
    cycles = results.get("cycles", [])
    lines.append("# Retrieval gate report\n")
    lines.append(f"Generated at: {results['generated_at']}\n")
    lines.append(
        f"Sweep: {len(per_config)} (model, bands) configs x {len(queries.all_concepts())} concepts. "
        f"Each config max-pools cosine across {len(cycles)} NAIP cycles per chip-location "
        f"(cycles: {', '.join(cycles)}). "
        f"Total chip-locations: {n_chips}. Prompt ensemble averages "
        f"{min(len(p) for p in queries.all_concepts().values())}-"
        f"{max(len(p) for p in queries.all_concepts().values())} synonyms per concept then renormalizes.\n"
    )

    # ---- section A: aggregate table ----
    findable_concepts = _findable_gt_concepts(per_config)
    n_findable = len(findable_concepts)
    n_total_gt = len(queries.GT_CONCEPTS)
    lines.append(
        f"## A. Aggregate over the {n_findable} findable GT concepts "
        f"(of {n_total_gt} total; cemetery + cabin excluded because every "
        f"feature is `findable_aerial=false` — see section D)\n"
    )
    lines.append("| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR |")
    lines.append("|---|---|---|---|---|---|---|")
    agg_rows = []
    for cfg in per_config:
        agg = {}
        for k in K_VALUES:
            agg[f"hit@{k}"] = _aggregate_metric(
                per_config, "findable", "hit@k", k, concepts=findable_concepts
            )[cfg]
        agg["recall@10"] = _aggregate_metric(
            per_config, "findable", "recall@k", 10, concepts=findable_concepts
        )[cfg]
        agg["mrr"] = _aggregate_metric(
            per_config, "findable", "mrr", concepts=findable_concepts
        )[cfg]
        agg_rows.append((cfg, agg))
    # also append random baseline as a synthetic config (over findable concepts)
    agg_rows.append(("RANDOM (closed-form mean)", {
        **{
            f"hit@{k}": float(np.mean([
                random_bl["per_concept"][c]["findable"]["hit@k"][k]
                for c in findable_concepts
            ]))
            for k in K_VALUES
        },
        "recall@10": float(np.mean([
            random_bl["per_concept"][c]["findable"]["recall@k"][10]
            for c in findable_concepts
        ])),
        "mrr": float(np.mean([
            random_bl["per_concept"][c]["findable"]["mrr"]
            for c in findable_concepts
        ])),
    }))
    for cfg, agg in agg_rows:
        lines.append(
            f"| {_config_label(cfg)} | "
            + " | ".join([_fmt_pct(agg[f'hit@{k}']) for k in K_VALUES])
            + f" | {_fmt_pct(agg['recall@10'])} | {_fmt_score(agg['mrr'])} |"
        )
    lines.append("")

    # winner pick
    best_cfg = max(
        (c for c, _ in agg_rows if c.startswith("RANDOM") is False),
        key=lambda c: dict(agg_rows)[c]["mrr"],
    )
    lines.append(f"**Best config by mean findable MRR: `{_config_label(best_cfg)}`**\n")

    # ---- section B: per-concept tables ----
    lines.append("## B. Per-concept results (findable GT)\n")
    for concept in queries.GT_CONCEPTS:
        nf = len(gt_findable.get(concept, set()))
        ns = len(gt_strict.get(concept, set()))
        lines.append(f"### {concept}")
        lines.append(f"n_gt findable: {nf}, n_gt strict: {ns}\n")
        lines.append("| config | hit@1 | hit@5 | hit@10 | hit@20 | recall@10 | MRR | top-1 score |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cfg, cres in per_config.items():
            f = cres[concept]["findable"]
            row = (
                f"| {_config_label(cfg)} | "
                f"{_fmt_pct(f['hit_at_k'].get(1, 0))} | "
                f"{_fmt_pct(f['hit_at_k'].get(5, 0))} | "
                f"{_fmt_pct(f['hit_at_k'].get(10, 0))} | "
                f"{_fmt_pct(f['hit_at_k'].get(20, 0))} | "
                f"{_fmt_pct(f['recall_at_k'].get(10, 0))} | "
                f"{_fmt_score(f['mrr'])} | "
                f"{_fmt_score(f['top_k_scores'][0])} |"
            )
            lines.append(row)
        # random
        rb = random_bl["per_concept"][concept]["findable"]
        lines.append(
            f"| RANDOM | "
            + " | ".join([_fmt_pct(rb['hit@k'][k]) for k in K_VALUES])
            + f" | {_fmt_pct(rb['recall@k'][10])} | {_fmt_score(rb['mrr'])} | n/a |"
        )
        lines.append("")

    # ---- section C: per-concept best-config pick ----
    lines.append("## C. Best (model, bands) per concept\n")
    lines.append("| concept | n_gt findable | best config | best hit@10 | best MRR |")
    lines.append("|---|---|---|---|---|")
    for concept in queries.GT_CONCEPTS:
        best = None
        best_mrr = -1.0
        for cfg, cres in per_config.items():
            mrr = cres[concept]["findable"]["mrr"]
            if mrr > best_mrr:
                best_mrr = mrr
                best = cfg
        f = per_config[best][concept]["findable"]
        lines.append(
            f"| {concept} | {len(gt_findable.get(concept, set()))} | {_config_label(best)} | "
            f"{_fmt_pct(f['hit_at_k'].get(10, 0))} | {_fmt_score(f['mrr'])} |"
        )
    lines.append("")

    # ---- section D: strict GT vs findable (what aerial-only leaves for LiDAR) ----
    lines.append("## D. Strict vs findable GT per concept (aerial-only gap)\n")
    lines.append(
        "When `n_gt_strict > n_gt_findable`, the extra features are flagged "
        "`findable_aerial = false` in the eval set (canopy-occluded). A "
        "persistent strict-hit@10 = 0 on those concepts is the signal that "
        "the LiDAR modality needs to recover them.\n"
    )
    lines.append("| concept | n_gt findable | n_gt strict | best findable hit@10 | best strict hit@10 | gap |")
    lines.append("|---|---|---|---|---|---|")
    for concept in queries.GT_CONCEPTS:
        nf = len(gt_findable.get(concept, set()))
        ns = len(gt_strict.get(concept, set()))
        best_f = max(per_config[c][concept]["findable"]["hit_at_k"].get(10, 0) for c in per_config)
        best_s = max(per_config[c][concept]["strict"]["hit_at_k"].get(10, 0) for c in per_config)
        gap = best_f - best_s
        gap_str = f"{gap:+.1%}" if gap != 0 else "0.0%"
        lines.append(
            f"| {concept} | {nf} | {ns} | {_fmt_pct(best_f)} | {_fmt_pct(best_s)} | {gap_str} |"
        )
    lines.append("")

    # ---- section E: no-GT concepts (qualitative only) ----
    lines.append("## E. No-GT concepts (visual top-K only)\n")
    tk_dir = topk_chips_dir(experiment_id)
    lines.append(
        f"No recall metric for these (no labeled feature in the eval set). "
        f"Open the thumbnail folders under "
        f"`{tk_dir.relative_to(tk_dir.parents[3])}/` to inspect:\n"
    )
    for concept in queries.NO_GT_CONCEPTS:
        lines.append(f"- **{concept}**")
        for cfg in per_config:
            d = topk_chip_dir(experiment_id, concept, *cfg.split("__"))
            lines.append(f"  - `{d.relative_to(d.parents[4])}`")
        lines.append("")

    # ---- section F: gate verdict (automated checks 1-5) ----
    lines.append("## F. Automated gate checks\n")
    waterfall_top1_best = max(
        per_config[c]["waterfall"]["findable"]["top_k_scores"][0]
        for c in per_config
    )
    waterfall_hit1_best = max(
        per_config[c]["waterfall"]["findable"]["hit_at_k"].get(1, 0)
        for c in per_config
    )
    # gate check #4 is over findable concepts (those where finding is even
    # possible in aerial). Cemetery + cabin are excluded since the eval set
    # flags them findable_aerial=false; they route to the LiDAR modality.
    in_scope_concepts_hit10 = {
        concept: max(
            per_config[c][concept]["findable"]["hit_at_k"].get(10, 0)
            for c in per_config
        )
        for concept in findable_concepts
    }
    n_hit10 = sum(1 for v in in_scope_concepts_hit10.values() if v == 1.0)
    hit10_target = max(1, int(round(0.7 * len(findable_concepts))))

    best_mrr_per_config = _aggregate_metric(
        per_config, "findable", "mrr", concepts=findable_concepts
    )
    rb_mean_mrr = random_bl["aggregate"]["findable"]["mrr"]
    # historical floor baseline kept here for sweeps that opt in to the
    # CLIP-B/32 baseline. Production runs (single production model) skip
    # the floor check vacuously — see check #5 below.
    floor_id = "clip-vit-b-32-openai"

    def _is_rs(cfg_key: str) -> bool:
        if cfg_key.startswith("ensemble:"):
            members = cfg_key.split("ensemble:")[1].split("__")[0].split("+")
            return any(m != floor_id for m in members)
        return cfg_key.split("__")[0] != floor_id

    def _is_floor(cfg_key: str) -> bool:
        return cfg_key.startswith(f"{floor_id}__")

    rs_mrrs = [v for cfg, v in best_mrr_per_config.items() if _is_rs(cfg)]
    floor_mrrs = [v for cfg, v in best_mrr_per_config.items() if _is_floor(cfg)]
    rs_best_mrr = max(rs_mrrs) if rs_mrrs else 0.0
    floor_best_mrr = max(floor_mrrs) if floor_mrrs else None  # None = not present

    checks = [
        (f"1. Pipeline end-to-end ({len(per_config)} configs + report rendered)", True,
         f"{len(per_config)} configs evaluated over {len(cycles)} cycles"),
        ("2. Best config aggregate hit@10 beats random", any(
            agg["hit@10"] > 5 * random_bl["aggregate"]["findable"]["hit@k"][10]
            for _, agg in agg_rows if not _.startswith("RANDOM")
        ),
         f"random hit@10 = {_fmt_pct(random_bl['aggregate']['findable']['hit@k'][10])}, "
         f"best = {_fmt_pct(max(agg['hit@10'] for _, agg in agg_rows if not _.startswith('RANDOM')))}"
        ),
        ("3. Waterfall hits @1 = 1.0 with the best config",
         waterfall_hit1_best == 1.0,
         f"best waterfall hit@1 = {waterfall_hit1_best}"),
        (f"4. >=70% of {len(findable_concepts)} findable in-scope concepts have findable hit@10 = 1.0",
         n_hit10 >= hit10_target,
         f"{n_hit10}/{len(findable_concepts)} findable concepts hit@10=1.0 "
         f"(target: >={hit10_target}): "
         + ", ".join(c for c, v in in_scope_concepts_hit10.items() if v == 1.0)),
        ("5. At least one RS-CLIP beats CLIP floor on mean MRR",
         (floor_best_mrr is None) or (rs_best_mrr > floor_best_mrr),
         (
             f"RS-CLIP best mean MRR = {_fmt_score(rs_best_mrr)}, "
             + (f"CLIP floor best = {_fmt_score(floor_best_mrr)}" if floor_best_mrr is not None
                else "(floor not in this run; check skipped)")
         )),
    ]
    lines.append("| check | result | note |")
    lines.append("|---|---|---|")
    for label, ok, note in checks:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} | {note} |")
    lines.append("")

    n_passed = sum(1 for _, ok, _ in checks if ok)
    if n_passed == 5:
        verdict = "**GO** - all 5 automated checks pass."
    elif n_passed >= 3 and checks[2][1]:
        verdict = (
            f"**ITERATE** - {n_passed}/5 automated checks pass. "
            f"Waterfall demo works; some in-scope concepts under-performing. "
            f"Manual gate decides whether to bring in LiDAR (S8/S9) or accept as-is."
        )
    else:
        verdict = (
            f"**KILL candidate** - {n_passed}/5 automated checks pass; the waterfall "
            f"demo {'works' if checks[2][1] else 'is broken'}. "
            f"Inspect thumbnails before calling the verdict."
        )
    lines.append(f"### Automated verdict: {verdict}\n")
    lines.append(
        "The final go / iterate / kill call is yours. Open the thumbnail folders "
        "(section E + per-concept best configs in C) to verify the metric reflects "
        "reality.\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the N0 retrieval gate harness.")
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help=(
            "Subset of (model, bands) configs as 'model_id__bands' tokens. "
            "Default = the YAML model_id on YAML bands (one config). To add "
            "a candidate to an A/B sweep, register it in embed.models.MODELS "
            "and pass it here."
        ),
    )
    parser.add_argument(
        "--no-thumbnails", action="store_true",
        help="Skip rendering top-K chip thumbnails (faster iteration).",
    )
    parser.add_argument(
        "--cycles", nargs="*", default=None,
        help="Subset of cycles; default = the YAML cycles.",
    )
    parser.add_argument(
        "--ensemble", default=None,
        help=(
            "Run the ensemble search across a comma-separated list of model "
            "ids registered in embed.models.MODELS. The first id is the primary "
            "(its argmax cycle is reported per chip). Pairs with --bands "
            "(default = YAML bands). Mutually exclusive with --configs."
        ),
    )
    parser.add_argument(
        "--bands", default=None, choices=ALL_BANDS,
        help="Bands for --ensemble (single value). Ignored when --configs is set.",
    )
    args = parser.parse_args()

    cfg = config.load_experiment(args.experiment)
    experiment_id = config.experiment_id_of(cfg)
    aoi_id = config.aoi_id_of(cfg)
    eval_set_id = config.eval_set_id_of(cfg)
    yaml_model = config.model_id_of(cfg)
    yaml_bands = config.bands_of(cfg)
    yaml_cycles = config.cycles_of(cfg)

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    chip_index = n0_retrieval.load_chip_index(experiment_id)
    eval_features = n0_retrieval.load_eval_features(eval_set_id)
    locations = n0_retrieval.chip_location_index(chip_index)
    n_chips = len(locations)

    gt_findable = n0_retrieval.build_ground_truth(chip_index, eval_features, findable_only=True)
    gt_strict = n0_retrieval.build_ground_truth(chip_index, eval_features, findable_only=False)

    cycles = args.cycles or yaml_cycles
    fallback_cycle = sorted(c["year"] for c in chip_index["cycles"])[-1]
    weights_manifest = json.loads(MODEL_WEIGHTS_MANIFEST.read_text())

    if args.ensemble and args.configs:
        raise SystemExit("--ensemble and --configs are mutually exclusive")

    bands = args.bands or yaml_bands

    ensemble_members: list[str] | None = None
    if args.ensemble:
        ensemble_members = [m.strip() for m in args.ensemble.split(",") if m.strip()]
        if len(ensemble_members) < 2:
            raise SystemExit("--ensemble needs at least 2 comma-separated model ids")
        configs = []  # not used in ensemble mode
    elif args.configs:
        configs = [tuple(c.split("__")) for c in args.configs]
    else:
        # default: the YAML's (model_id, bands)
        configs = [(yaml_model, yaml_bands)]

    if ensemble_members:
        print(
            f"=== N0 retrieval: ENSEMBLE {ensemble_members} / {bands} "
            f"over {len(cycles)} cycles ==="
        )
    else:
        print(f"=== N0 retrieval: {len(configs)} configs over {len(cycles)} cycles ===")
    print(f"GT findable: {sum(len(v) for v in gt_findable.values())} chip-locations across "
          f"{sum(1 for v in gt_findable.values() if v)} concepts")
    print(f"GT strict:   {sum(len(v) for v in gt_strict.values())} chip-locations across "
          f"{sum(1 for v in gt_strict.values() if v)} concepts")
    print()

    per_config: dict[str, dict] = {}
    t0 = time.time()
    if ensemble_members:
        # ensemble stays on the in-memory numpy path. The MVP has one
        # registered model, so this code path is exercised only when the
        # user has manually re-registered candidates and kept their .npy
        # files. A DB-backed ensemble lands at R2 when multi-model is real.
        cfg_key = _ensemble_key(ensemble_members, bands)
        combo_out = evaluate_ensemble(
            experiment_id=experiment_id, aoi_id=aoi_id, fallback_cycle=fallback_cycle,
            member_model_ids=ensemble_members, bands=bands, cycles=cycles,
            concepts=queries.all_concepts(),
            gt_findable=gt_findable, gt_strict=gt_strict,
            locations=locations, device=device, weights_manifest=weights_manifest,
        )
        per_config[cfg_key] = combo_out
    else:
        chip_loc_to_idx = n0_retrieval.chip_location_to_index_map(chip_index)
        with vs_connect() as db_conn:
            for model_id, b in configs:
                cfg_key = _config_key(model_id, b)
                weights_entry = weights_manifest["models"][model_id]
                combo_out = evaluate_one_combo(
                    experiment_id=experiment_id, aoi_id=aoi_id, fallback_cycle=fallback_cycle,
                    model_id=model_id, bands=b, cycles=cycles,
                    concepts=queries.all_concepts(),
                    gt_findable=gt_findable, gt_strict=gt_strict,
                    locations=locations, device=device, weights_entry=weights_entry,
                    chip_loc_to_idx=chip_loc_to_idx, db_conn=db_conn,
                )
                per_config[cfg_key] = combo_out
    dt = time.time() - t0

    random_baseline = build_random_baseline(n_chips, gt_findable, gt_strict)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "n_chips": n_chips,
        "cycles": cycles,
        "n_configs": len(per_config),
        "per_config": per_config,
        "random_baseline": random_baseline,
        "gt_sizes": {
            "findable": {c: len(v) for c, v in gt_findable.items()},
            "strict":   {c: len(v) for c, v in gt_strict.items()},
        },
        "elapsed_s": round(dt, 2),
    }
    results_path = n0_results_json(experiment_id)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {results_path}")
    write_markdown_report(experiment_id, results, gt_findable, gt_strict)


if __name__ == "__main__":
    main()
