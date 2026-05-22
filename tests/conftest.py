"""Integration-test fixtures for the vector store.

Each session creates one ephemeral schema in the dev DB and applies
the production DDL into it. Per-test cleanup is a TRUNCATE, so the
HNSW + GIST indexes are paid for once.

The DSN is read from TEST_DATABASE_URL, falling back to
TERRA_QUERY_DATABASE_URL, falling back to the docker-compose default.
If the resolved DB is not reachable, the integration tests skip.
"""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from pgvector.psycopg import register_vector

from terra_query.core.paths import VECTOR_STORE_SCHEMA_SQL
from terra_query.vector_store.db import DEFAULT_DSN, DSN_ENV_VAR

TEST_DSN_ENV = "TEST_DATABASE_URL"


def _resolve_test_dsn() -> str:
    return (
        os.environ.get(TEST_DSN_ENV)
        or os.environ.get(DSN_ENV_VAR)
        or DEFAULT_DSN
    )


def _can_connect(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_dsn() -> str:
    dsn = _resolve_test_dsn()
    if not _can_connect(dsn):
        pytest.skip(f"vector-store integration: DB not reachable at {dsn}")
    return dsn


@pytest.fixture(scope="session")
def test_schema(db_dsn: str):
    """Create + drop a unique schema for this test session."""
    schema = f"vstest_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}", public')
            cur.execute(VECTOR_STORE_SCHEMA_SQL.read_text())
    try:
        yield schema
    finally:
        with psycopg.connect(db_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def db_conn(db_dsn: str, test_schema: str):
    """Yield a fresh connection scoped to the test schema; truncates before yielding."""
    with psycopg.connect(db_dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f'SET search_path TO "{test_schema}", public')
            cur.execute("TRUNCATE chip_embeddings")
        conn.commit()
        yield conn
