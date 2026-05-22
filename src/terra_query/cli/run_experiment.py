"""CLI: run every pipeline stage in order, end-to-end.

The umbrella CLI for "I changed something in the YAML, now bring
everything up to date." Each stage is already idempotent on its own,
so re-running with no YAML change is a sequence of no-ops.

Stages (in order):

  1. reproject_inputs   — AOI + eval set into the working CRS
  2. discover_aerial    — STAC pick, write NAIP / S2 manifests
  3. fetch_naip         — per-cycle COG download
  4. fetch_sentinel2    — per-scene COG download
  5. build_chip_index   — chip grid + eval-chip PNGs + grid overview
  6. fetch_weights      — model weights download
  7. embed_chips        — sweep over bands x cycles
  8. init_db            — apply schema (idempotent CREATE IF NOT EXISTS)
  9. load_embeddings    — upsert into chip_embeddings
  10. run_n0_retrieval  — regenerate the gate report

NOT the R2 graph-walking CLI: this is a sequential walker. Each stage
decides whether its own work is needed. The umbrella exists to give
"edit YAML, run one command, everything cascades" UX without R2's
content-addressed staleness machinery.

Prereq: the vector store must be reachable (`docker compose up -d`).

Examples:

    # full pipeline, default config
    uv run python -m terra_query.cli.run_experiment

    # plan only
    uv run python -m terra_query.cli.run_experiment --dry-run

    # only re-embed + reload + regen gate
    uv run python -m terra_query.cli.run_experiment --from embed_chips

    # one stage
    uv run python -m terra_query.cli.run_experiment --only build_chip_index
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from terra_query.core import config


@dataclass(frozen=True)
class Stage:
    name: str
    module: str           # the `python -m <module>` target
    description: str
    pass_experiment: bool # whether to pass --experiment <path> to the subprocess


STAGES: list[Stage] = [
    Stage("reproject_inputs",  "terra_query.ingest.cli.reproject_inputs",
          "Reproject AOI + eval set to working CRS",      True),
    Stage("discover_aerial",   "terra_query.ingest.cli.discover_aerial",
          "STAC discovery: write NAIP + S2 manifests",    True),
    Stage("fetch_naip",        "terra_query.ingest.cli.fetch_naip",
          "Download NAIP COGs per cycle",                 True),
    Stage("fetch_sentinel2",   "terra_query.ingest.cli.fetch_sentinel2",
          "Download Sentinel-2 winter COGs",              True),
    Stage("build_chip_index",  "terra_query.ingest.cli.build_chip_index",
          "Cut chip grid + eval-chip PNGs + overview",    True),
    Stage("fetch_weights",     "terra_query.embed.cli.fetch_weights",
          "Download model weights",                       True),
    Stage("embed_chips",       "terra_query.embed.cli.embed_chips",
          "Embed every chip (bands x cycles)",            True),
    Stage("init_db",           "terra_query.vector_store.cli.init_db",
          "Apply vector-store schema (idempotent)",       False),
    Stage("load_embeddings",   "terra_query.vector_store.cli.load_embeddings",
          "Load embeddings into chip_embeddings",         True),
    Stage("run_n0_retrieval",  "terra_query.eval.cli.run_n0_retrieval",
          "Regenerate the N0 gate report",                True),
]

STAGE_NAMES = [s.name for s in STAGES]


def _stage(name: str) -> Stage:
    for s in STAGES:
        if s.name == name:
            return s
    raise SystemExit(
        f"unknown stage {name!r}. Known: {STAGE_NAMES}"
    )


def _pick_range(args) -> list[Stage]:
    if args.only:
        return [_stage(args.only)]
    start = STAGE_NAMES.index(args.from_stage) if args.from_stage else 0
    stop_inclusive = (
        STAGE_NAMES.index(args.to_stage) if args.to_stage else len(STAGE_NAMES) - 1
    )
    if start > stop_inclusive:
        raise SystemExit(
            f"--from {args.from_stage!r} is after --to {args.to_stage!r}"
        )
    return STAGES[start : stop_inclusive + 1]


def _build_argv(stage: Stage, experiment_path: Path) -> list[str]:
    argv = [sys.executable, "-m", stage.module]
    if stage.pass_experiment:
        argv += ["--experiment", str(experiment_path)]
    return argv


def _format_plan(experiment_path: Path, picked: list[Stage]) -> str:
    lines = [
        "=" * 64,
        "run_experiment plan",
        "=" * 64,
        f"  experiment YAML : {experiment_path}",
        f"  stages          : {len(picked)} of {len(STAGES)}",
        "",
    ]
    for i, st in enumerate(picked, start=1):
        argv = _build_argv(st, experiment_path)
        lines.append(f"  {i:>2}. {st.name:20s} - {st.description}")
        lines.append(f"        {' '.join(argv[2:])}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan, exit 0; no stages run.",
    )
    parser.add_argument(
        "--from", dest="from_stage", choices=STAGE_NAMES, default=None,
        help="Start at this stage (skip earlier ones).",
    )
    parser.add_argument(
        "--to", dest="to_stage", choices=STAGE_NAMES, default=None,
        help="Stop after this stage.",
    )
    parser.add_argument(
        "--only", choices=STAGE_NAMES, default=None,
        help="Run just this single stage. Overrides --from / --to.",
    )
    args = parser.parse_args()

    # resolve the experiment path explicitly so the subprocess invocations all
    # point at the same file (no surprises from differing env / cwd)
    cfg = config.load_experiment(args.experiment)
    experiment_path = config.source_path_of(cfg)

    picked = _pick_range(args)
    print(_format_plan(experiment_path, picked))

    if args.dry_run:
        print("\n--dry-run: no stages will run.")
        return 0

    t_all = time.time()
    for i, st in enumerate(picked, start=1):
        print(f"\n[{i}/{len(picked)}] {st.name}")
        argv = _build_argv(st, experiment_path)
        t0 = time.time()
        r = subprocess.run(argv)
        dt = time.time() - t0
        if r.returncode != 0:
            print(
                f"\n[{st.name}] FAILED after {dt:.1f}s with exit code "
                f"{r.returncode}. Stopping pipeline."
            )
            return r.returncode
        print(f"  ({st.name} done in {dt:.1f}s)")

    print(f"\npipeline complete in {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
