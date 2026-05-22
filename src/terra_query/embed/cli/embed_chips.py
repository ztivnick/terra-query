"""CLI: embed chips for one or more (model, bands, cycle) cells.

Idempotent. Skips a cell whose .npy + sidecar are already up to date
relative to the chip index + the weights manifest entry.

Defaults: the experiment YAML's `model_id` x `bands` x `cycles`. To
sweep a candidate model, register it in `embed.models.MODELS` and pass
`--models <id>`.

Examples:
  # production sweep (default): YAML model x YAML bands x YAML cycles
  python -m terra_query.embed.cli.embed_chips
  # just one cycle for a quick smoke test
  python -m terra_query.embed.cli.embed_chips --cycles 2022
  # one specific cell
  python -m terra_query.embed.cli.embed_chips \\
      --models <some-registered-id> --bands rgb --cycles 2022
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from terra_query.core import config
from terra_query.core.paths import chip_index_json
from terra_query.embed import models
from terra_query.embed.embed_chips import DEFAULT_NUM_WORKERS, embed_cycles_for_combo

ALL_BANDS = ["rgb", "cir"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed NAIP chips per (model, bands, cycle).")
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model ids; default = the YAML model_id.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=None,
        choices=ALL_BANDS,
        help="Subset of band combos; default = the YAML bands.",
    )
    parser.add_argument(
        "--cycles",
        nargs="*",
        default=None,
        help='Subset of cycles or "all"; default = the YAML cycles.',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if up-to-date.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-model default batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "DataLoader worker processes for chip reads + preprocess (default "
            f"{DEFAULT_NUM_WORKERS}). 0 = single-process (deprecated baseline)."
        ),
    )
    args = parser.parse_args()

    cfg = config.load_experiment(args.experiment)
    experiment_id = config.experiment_id_of(cfg)
    aoi_id = config.aoi_id_of(cfg)
    yaml_model = config.model_id_of(cfg)
    yaml_bands = config.bands_of(cfg)
    yaml_cycles = config.cycles_of(cfg)

    targets_models = args.models or [yaml_model]
    targets_bands = args.bands or [yaml_bands]
    if args.cycles is None:
        targets_cycles = yaml_cycles
    elif args.cycles == ["all"]:
        targets_cycles = yaml_cycles
    else:
        targets_cycles = list(args.cycles)

    idx = json.loads(chip_index_json(experiment_id).read_text())
    available_years = {c["year"] for c in idx["cycles"]}
    bad = [y for y in targets_cycles if y not in available_years]
    if bad:
        raise SystemExit(f"cycles not in chip index: {bad}; available: {sorted(available_years)}")

    n_cells = len(targets_models) * len(targets_bands) * len(targets_cycles)
    print(f"=== embed sweep: {n_cells} cells "
          f"({len(targets_models)} models x {len(targets_bands)} bands x "
          f"{len(targets_cycles)} cycles) ===")
    for m in targets_models:
        for b in targets_bands:
            for c in targets_cycles:
                print(f"  - {m} / {b} / {c}")
    print()

    t0_all = time.time()
    sidecars: list[dict] = []
    for m in targets_models:
        for b in targets_bands:
            batch = args.batch_size or models.spec(m).mps_batch_size
            try:
                scs = embed_cycles_for_combo(
                    experiment_id, aoi_id, targets_cycles, m, b,
                    batch_size=batch, force=args.force,
                    num_workers=args.num_workers,
                )
                sidecars.extend(scs)
            except Exception as e:
                print(f"ERROR on combo ({m}, {b}): {type(e).__name__}: {e}")
                raise
    dt_all = time.time() - t0_all
    print(f"\n=== done: {n_cells} cells in {dt_all:.1f}s ({dt_all/60:.1f} min) ===")


if __name__ == "__main__":
    main()
