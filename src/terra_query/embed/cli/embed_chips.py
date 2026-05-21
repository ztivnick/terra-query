"""CLI: embed chips for one or more (model, bands, cycle) cells.

Idempotent. Skips a cell whose .npy + sidecar are already up to date
relative to the chip index + the weights manifest entry.

Defaults: the production model (`models.PRODUCTION_MODEL_ID`) on RGB
across the 6 NAIP cycles. To run a candidate model or CIR, register the
model in `embed.models.MODELS` and pass `--models <id>` / `--bands cir`.

Examples:
  # production sweep (default): production model x rgb x 6 cycles
  python -m terra_query.embed.cli.embed_chips
  # just one cycle for a quick smoke test
  python -m terra_query.embed.cli.embed_chips --cycles 2022
  # one specific cell
  python -m terra_query.embed.cli.embed_chips \\
      --models georsclip-vit-l-14-336 --bands rgb --cycles 2022
"""

from __future__ import annotations

import argparse
import json
import time

from terra_query.core.paths import CHIP_INDEX_JSON
from terra_query.embed import models
from terra_query.embed.embed_chips import DEFAULT_NUM_WORKERS, embed_cycles_for_combo

# per-model MPS batch ceiling. ViT-L-14-336 has ~2.25x more transformer
# tokens per image than the 224 variant; 16 keeps headroom on 32 GB unified
# memory while staying GPU-saturated.
BATCH_SIZE_BY_MODEL = {
    "georsclip-vit-l-14-336": 16,
}
DEFAULT_BATCH_SIZE = 16

ALL_CYCLES = ["2012", "2014", "2016", "2018", "2020", "2022"]
ALL_BANDS = ["rgb", "cir"]
DEFAULT_BANDS = ["rgb"]  # production embedding is RGB-only; CIR is opt-in


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed NAIP chips per (model, bands, cycle).")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model ids; default = the production model.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=None,
        choices=ALL_BANDS,
        help="Subset of band combos; default = rgb only.",
    )
    parser.add_argument(
        "--cycles",
        nargs="*",
        default=None,
        help='Subset of NAIP cycles (e.g. 2022) or "all"; default = all 6.',
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

    targets_models = args.models or models.model_ids()
    targets_bands = args.bands or DEFAULT_BANDS
    if args.cycles is None or args.cycles == ["all"]:
        targets_cycles = ALL_CYCLES
    else:
        targets_cycles = list(args.cycles)

    idx = json.loads(CHIP_INDEX_JSON.read_text())
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
            batch = args.batch_size or BATCH_SIZE_BY_MODEL.get(m, DEFAULT_BATCH_SIZE)
            try:
                scs = embed_cycles_for_combo(
                    targets_cycles, m, b,
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
