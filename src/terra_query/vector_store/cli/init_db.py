"""CLI: apply the vector-store schema to whichever DB the DSN points at.

Idempotent: every DDL statement uses IF NOT EXISTS.
"""

from __future__ import annotations

import argparse

from terra_query.vector_store.db import connect, get_dsn
from terra_query.vector_store.schema import apply_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the vector-store schema.")
    parser.add_argument(
        "--dsn",
        default=None,
        help="Override the DSN; defaults to TERRA_QUERY_DATABASE_URL or localhost.",
    )
    args = parser.parse_args()

    dsn = args.dsn or get_dsn()
    print(f"connecting to {dsn}")
    with connect(dsn) as conn:
        apply_schema(conn)
    print("schema applied")


if __name__ == "__main__":
    main()
