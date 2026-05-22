"""Tests for terra_query.core.storage."""

from __future__ import annotations

import pytest

from terra_query.core.storage import LocalFilesystemStorage, Storage


def test_local_storage_roundtrip(tmp_path):
    s = LocalFilesystemStorage(root=tmp_path / "blobs", url_prefix="/thumbnails/")
    assert not s.exists("a.png")
    s.write_bytes("a.png", b"hello")
    assert s.exists("a.png")
    assert s.read_bytes("a.png") == b"hello"


def test_local_storage_url_for(tmp_path):
    s = LocalFilesystemStorage(root=tmp_path, url_prefix="/thumbnails/")
    assert s.url_for("a.png") == "/thumbnails/a.png"


def test_url_prefix_normalized(tmp_path):
    s = LocalFilesystemStorage(root=tmp_path, url_prefix="/x")
    assert s.url_for("y") == "/x/y"


def test_local_storage_creates_parent_dir(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    s = LocalFilesystemStorage(root=root, url_prefix="/t/")
    s.write_bytes("a.png", b"x")
    assert (root / "a.png").exists()


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "../x"])
def test_local_storage_rejects_traversal(tmp_path, bad):
    s = LocalFilesystemStorage(root=tmp_path, url_prefix="/t/")
    with pytest.raises(ValueError):
        s.write_bytes(bad, b"x")
    with pytest.raises(ValueError):
        s.url_for(bad)


def test_storage_is_abstract():
    with pytest.raises(TypeError):
        Storage()  # type: ignore[abstract]
