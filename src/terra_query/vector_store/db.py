"""Connection helpers + DSN resolution.

DSN comes from the env var `TERRA_QUERY_DATABASE_URL`, defaulting to
the local docker-compose Postgres on port 5433. The DSN intentionally
stays outside the experiment YAML (R1 decision): it's a 12-factor
secret/credential and a deployment-target setting, not a per-experiment
knob. Production deploys override the env var.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

DSN_ENV_VAR = "TERRA_QUERY_DATABASE_URL"
DEFAULT_DSN = "postgresql://terra:terra@localhost:5433/terra_query"


def get_dsn() -> str:
    """Return the DSN to connect with.

    Reads `TERRA_QUERY_DATABASE_URL` if set, else the local-dev default.
    """
    return os.environ.get(DSN_ENV_VAR, DEFAULT_DSN)


@contextmanager
def connect(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Open a Postgres connection with pgvector type adaptation registered.

    Caller-managed: the context manager commits on clean exit and rolls
    back on exception, then closes the connection. For the personal-scale
    MVP a single connection per operation is fine; a pool is unnecessary.
    """
    resolved = dsn or get_dsn()
    with psycopg.connect(resolved) as conn:
        register_vector(conn)
        yield conn
