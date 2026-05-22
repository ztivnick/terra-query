"""Tests for /thumbnails/{key} (cache-hit path + validation).

The cache-miss path generates a PNG by reading the source NAIP COG;
that's exercised manually at S7 gate. Here we cover:
- bad key shapes 400
- wrong-band key 400
- cache hit serves bytes from the configured Storage
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from terra_query.api.app import create_app


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TERRA_QUERY_SKIP_MODEL_LOAD", "1")
    app = create_app()
    with TestClient(app) as c:
        # swap storage to one rooted at tmp_path so the test doesn't touch
        # the real cache dir
        from terra_query.core.storage import LocalFilesystemStorage

        c.app.state.storage = LocalFilesystemStorage(
            root=tmp_path / "thumbs", url_prefix="/thumbnails/"
        )
        yield c


def test_thumbnail_bad_extension(client):
    r = client.get("/thumbnails/foo.jpg")
    assert r.status_code == 400


def test_thumbnail_bad_shape(client):
    # missing the cycle / bands suffixes
    r = client.get("/thumbnails/r020_c020.png")
    assert r.status_code == 400


def test_thumbnail_wrong_bands(client):
    # config bands="rgb"; request asks for cir
    r = client.get("/thumbnails/r020_c020__2022__cir.png")
    assert r.status_code == 400


def test_thumbnail_cache_hit(client):
    # pre-write a PNG to the storage so the route never touches the COG
    fake_png_header = b"\x89PNG\r\n\x1a\n" + b"x" * 32
    key = "r020_c020__2022__rgb.png"
    client.app.state.storage.write_bytes(key, fake_png_header)

    r = client.get(f"/thumbnails/{key}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == fake_png_header
