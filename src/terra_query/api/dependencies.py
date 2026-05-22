"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi import Request

from terra_query.vector_store.db import connect


def get_app_state(request: Request) -> Any:
    return request.app.state


def get_db_conn(request: Request) -> Iterator[Any]:
    with connect() as conn:
        yield conn
