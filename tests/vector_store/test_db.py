"""Unit tests for DSN resolution. No DB required."""

from __future__ import annotations

from terra_query.vector_store import db


def test_default_dsn_when_env_unset(monkeypatch):
    monkeypatch.delenv(db.DSN_ENV_VAR, raising=False)
    assert db.get_dsn() == db.DEFAULT_DSN


def test_env_var_overrides_default(monkeypatch):
    custom = "postgresql://other:other@db.example:5432/other"
    monkeypatch.setenv(db.DSN_ENV_VAR, custom)
    assert db.get_dsn() == custom


def test_default_dsn_points_at_local_compose_port():
    # the docker-compose binds to 5433; matching default avoids surprises.
    assert "localhost:5433" in db.DEFAULT_DSN
    assert "/terra_query" in db.DEFAULT_DSN
