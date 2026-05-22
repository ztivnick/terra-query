"""CLI: swap the production model end-to-end.

Two flows, one code path:

- **YAML-detect (default)**: edit the YAML's `model_id` to bump the target,
  then run `python -m terra_query.embed.cli.swap_model`. The CLI reads
  the YAML's `model_id` as the target; identifies the "current" model(s)
  from on-disk embeddings + DB rows for this experiment; if they
  differ, orchestrates the swap.

- **`--to <new_model_id>`**: the explicit override. Useful for A/B
  candidates without hand-editing YAML. On success the YAML's
  `model_id` is rewritten to point at `--to`.

Orchestration order:
  1. fetch_weights         — download / verify the target's weights
  2. embed_chips           — embed every chip with the target (bands x cycles)
  3. load_embeddings       — upsert into chip_embeddings
  4. run_n0_retrieval      — regenerate the gate report + thumbnails
  5. purge old artifacts   — disk + `DELETE FROM chip_embeddings WHERE model_id = ...`
  6. rewrite YAML model_id (only when --to was provided)

Safety: the destructive purge is always behind a confirm prompt or
`--yes`. `--dry-run` prints the plan with no side effects (no fetch,
no embed, no DB write, no YAML rewrite).

Examples:

    # bump YAML model_id by hand, then:
    uv run python -m terra_query.embed.cli.swap_model

    # explicit candidate (A/B), see the plan without doing anything:
    uv run python -m terra_query.embed.cli.swap_model --to remoteclip-vit-l-14 --dry-run

    # run the swap, skipping the confirm:
    uv run python -m terra_query.embed.cli.swap_model --to remoteclip-vit-l-14 --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from terra_query.core import config
from terra_query.core.paths import (
    MODEL_WEIGHTS_MANIFEST,
    embeddings_dir,
    embeddings_json,
    embeddings_npy,
    model_weights_dir,
    topk_chips_dir,
)
from terra_query.embed import models


def _registered_models() -> list[str]:
    return sorted(models.MODELS.keys())


def _disk_models(experiment_id: str, bands: str) -> set[str]:
    """Model ids present on disk under this experiment's embeddings dir."""
    d = embeddings_dir(experiment_id)
    if not d.exists():
        return set()
    found: set[str] = set()
    for p in d.glob(f"*__{bands}__*.npy"):
        # filename: <model>__<bands>__<cycle>.npy
        parts = p.stem.split("__")
        if len(parts) == 3:
            found.add(parts[0])
    return found


def _db_models(bands: str) -> set[str]:
    """Distinct model_ids in chip_embeddings (this bands). Empty set if DB unreachable."""
    try:
        from terra_query.vector_store.db import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT model_id FROM chip_embeddings WHERE bands = %s",
                (bands,),
            )
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        print(f"[warn] DB unreachable ({e!s}); DB-side model inventory unavailable")
        return set()


def _purge_paths_for(experiment_id: str, model_id: str, bands: str,
                     cycles: list[str]) -> list[Path]:
    """The list of paths that get deleted when purging this model."""
    paths: list[Path] = []
    for c in cycles:
        paths.append(embeddings_npy(experiment_id, model_id, bands, c))
        paths.append(embeddings_json(experiment_id, model_id, bands, c))
    paths.append(model_weights_dir(model_id))
    # thumbnails: every subdir whose name contains __<model_id>__
    tk_root = topk_chips_dir(experiment_id)
    if tk_root.exists():
        for sub in tk_root.iterdir():
            if sub.is_dir() and f"__{model_id}__" in sub.name:
                paths.append(sub)
    return paths


def _format_plan(
    experiment_id: str,
    aoi_id: str,
    bands: str,
    cycles: list[str],
    target: str,
    current: set[str],
    to_purge: list[str],
    will_rewrite_yaml: bool,
    yaml_path: Path,
) -> str:
    lines = [
        "=" * 64,
        "swap_model plan",
        "=" * 64,
        f"  experiment_id    : {experiment_id}",
        f"  aoi_id           : {aoi_id}",
        f"  bands            : {bands}",
        f"  cycles           : {cycles}",
        f"  target model     : {target}",
        f"  current model(s) : {sorted(current) or '<none>'}",
        f"  to purge         : {to_purge or '<none>'}",
        f"  YAML path        : {yaml_path}",
        f"  will rewrite YAML: {will_rewrite_yaml}",
        "",
        "actions (in order):",
        "  1. fetch_weights for target",
        f"  2. embed_chips ({len(cycles)} cycle(s)) with target",
        "  3. load_embeddings into chip_embeddings",
        "  4. run_n0_retrieval (regenerate gate)",
    ]
    if to_purge:
        for m in to_purge:
            paths = _purge_paths_for(experiment_id, m, bands, cycles)
            lines.append(f"  5. purge artifacts for {m}:")
            for p in paths:
                exists = "exists" if p.exists() else "(not present)"
                lines.append(f"       {p} {exists}")
    if will_rewrite_yaml:
        lines.append(f"  6. rewrite YAML model_id -> {target}")
    lines.append("=" * 64)
    return "\n".join(lines)


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _fetch_weights_target(target: str) -> None:
    """Fetch + verify weights for `target`, write the manifest."""
    from terra_query.embed.cli.fetch_weights import (
        _fetch_one,
        _load_manifest,
        _save_manifest,
    )

    manifest = _load_manifest()
    entry = _fetch_one(target, force=False, manifest=manifest, verify_load=True)
    manifest["models"][target] = entry
    _save_manifest(manifest)


def _embed_target(experiment_id: str, aoi_id: str, target: str,
                  bands: str, cycles: list[str]) -> None:
    """Embed the chip set with `target` across (bands x cycles)."""
    from terra_query.embed.embed_chips import (
        DEFAULT_NUM_WORKERS,
        embed_cycles_for_combo,
    )

    batch = models.spec(target).mps_batch_size
    embed_cycles_for_combo(
        experiment_id, aoi_id, cycles, target, bands,
        batch_size=batch, force=False, num_workers=DEFAULT_NUM_WORKERS,
    )


def _load_target(experiment_id: str, target: str, bands: str,
                 cycles: list[str]) -> None:
    from terra_query.vector_store.db import connect
    from terra_query.vector_store.loader import load_all

    with connect() as conn:
        load_all(conn, experiment_id, target, bands, cycles)


def _regenerate_gate(experiment_yaml: Path) -> None:
    """Re-run the N0 gate harness as a subprocess (its main() owns argparse)."""
    cmd = [sys.executable, "-m", "terra_query.eval.cli.run_n0_retrieval",
           "--experiment", str(experiment_yaml)]
    subprocess.run(cmd, check=True)


def _purge_model(experiment_id: str, model_id: str, bands: str,
                 cycles: list[str]) -> None:
    """Delete disk artifacts + DB rows for `model_id`. Best-effort: never fatal."""
    for p in _purge_paths_for(experiment_id, model_id, bands, cycles):
        if not p.exists():
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            print(f"  rm {p}")
        except OSError as e:
            print(f"  [warn] could not remove {p}: {e}")
    # DB rows
    try:
        from terra_query.vector_store.db import connect

        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chip_embeddings WHERE model_id = %s",
                (model_id,),
            )
            n = cur.rowcount
            conn.commit()
            print(f"  DELETE FROM chip_embeddings WHERE model_id = {model_id!r} -> {n} rows")
    except Exception as e:
        print(f"  [warn] DB purge for {model_id} failed: {e}")


def _rewrite_yaml_model_id(yaml_path: Path, new_model_id: str) -> None:
    """Replace the YAML's `model_id` with `new_model_id`, preserving the rest."""
    with yaml_path.open("r") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise SystemExit(f"{yaml_path}: not a YAML mapping; cannot rewrite")
    doc["model_id"] = new_model_id
    with yaml_path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"  rewrote {yaml_path}: model_id -> {new_model_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    parser.add_argument(
        "--to", dest="to_model", default=None,
        help="Target model id (overrides YAML model_id). If set and the swap "
             "succeeds, the YAML is rewritten to point at this id.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan, exit 0; no fetch / embed / load / regen / purge / "
             "YAML rewrite.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the destructive-action confirm prompt.",
    )
    args = parser.parse_args()

    cfg = config.load_experiment(args.experiment)
    experiment_id = config.experiment_id_of(cfg)
    aoi_id = config.aoi_id_of(cfg)
    bands = config.bands_of(cfg)
    cycles = config.cycles_of(cfg)
    yaml_model = config.model_id_of(cfg)
    yaml_path = config.source_path_of(cfg)

    target = args.to_model or yaml_model

    if target not in models.MODELS:
        raise SystemExit(
            f"model id {target!r} is not registered in embed.models.MODELS. "
            f"Registered: {_registered_models()}. "
            f"Add a ModelSpec entry to MODELS{{}} first."
        )

    # what's currently on disk + in DB for this experiment / bands?
    on_disk = _disk_models(experiment_id, bands)
    in_db = _db_models(bands)
    current = on_disk | in_db

    if target in current and len(current) == 1:
        print(f"no-op (target == current): {target} is the only model present.")
        return 0

    to_purge = sorted(current - {target})
    will_rewrite_yaml = args.to_model is not None and yaml_model != target

    print(_format_plan(
        experiment_id, aoi_id, bands, cycles, target,
        current, to_purge, will_rewrite_yaml, yaml_path,
    ))

    if args.dry_run:
        print("\n--dry-run: no side effects.")
        return 0

    if not args.yes:
        if not _confirm("Proceed?"):
            print("aborted.")
            return 1

    # 1. fetch
    print("\n[1/6] fetch_weights ...")
    _fetch_weights_target(target)

    # 2. embed
    print("\n[2/6] embed_chips ...")
    _embed_target(experiment_id, aoi_id, target, bands, cycles)

    # 3. load
    print("\n[3/6] load_embeddings ...")
    _load_target(experiment_id, target, bands, cycles)

    # 4. regenerate gate
    print("\n[4/6] run_n0_retrieval ...")
    _regenerate_gate(yaml_path)

    # 5. purge old
    if to_purge:
        print(f"\n[5/6] purge {to_purge} ...")
        for m in to_purge:
            _purge_model(experiment_id, m, bands, cycles)
    else:
        print("\n[5/6] nothing to purge.")

    # 6. rewrite YAML
    if will_rewrite_yaml:
        print(f"\n[6/6] rewrite YAML model_id -> {target}")
        _rewrite_yaml_model_id(yaml_path, target)
    else:
        print("\n[6/6] YAML already at target; no rewrite needed.")

    print("\nswap complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
