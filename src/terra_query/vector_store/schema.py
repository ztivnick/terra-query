"""Apply the vector-store schema to a connected Postgres."""

from __future__ import annotations

import psycopg

from terra_query.core.paths import VECTOR_STORE_SCHEMA_SQL


def apply_schema(conn: psycopg.Connection) -> None:
    """Run the checked-in schema.sql against `conn`. Idempotent."""
    sql = VECTOR_STORE_SCHEMA_SQL.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
