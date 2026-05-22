"""CLI: load per-cycle embedding .npy files into chip_embeddings.

Defaults to the experiment YAML's `model_id` / `bands` / `cycles`.
Idempotent: re-running upserts on the primary key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from terra_query.core import config
from terra_query.vector_store.db import connect, get_dsn
from terra_query.vector_store.loader import _load_chip_index, load_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Load embeddings into chip_embeddings.")
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    parser.add_argument("--model", default=None, help="Override YAML model_id.")
    parser.add_argument("--bands", default=None, help="Override YAML bands.")
    parser.add_argument(
        "--cycles",
        nargs="*",
        default=None,
        help="Subset of cycles; default = all cycles in chip_index.",
    )
    parser.add_argument("--dsn", default=None)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing rows for this (model, bands) before loading.",
    )
    args = parser.parse_args()

    cfg = config.load_experiment(args.experiment)
    experiment_id = config.experiment_id_of(cfg)
    model_id = args.model or config.model_id_of(cfg)
    bands = args.bands or config.bands_of(cfg)

    chip_index = _load_chip_index(experiment_id)
    all_cycles = [c["year"] for c in chip_index["cycles"]]
    cycles = args.cycles if args.cycles is not None else all_cycles

    dsn = args.dsn or get_dsn()
    print(f"connecting to {dsn}")
    with connect(dsn) as conn:
        if args.clear:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chip_embeddings WHERE model_id = %s AND bands = %s",
                    (model_id, bands),
                )
                deleted = cur.rowcount
            conn.commit()
            print(f"cleared {deleted} prior rows for ({model_id}, {bands})")

        results = load_all(conn, experiment_id, model_id, bands, cycles)

    total = sum(results.values())
    print(f"loaded {total} rows across {len(results)} cycles")


if __name__ == "__main__":
    main()
