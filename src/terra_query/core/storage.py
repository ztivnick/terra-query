"""Binary blob store keyed by string. Concrete backends translate keys
into whatever addresses the underlying store uses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def url_for(self, key: str) -> str: ...


class LocalFilesystemStorage(Storage):
    """Stores blobs under `<root>/<key>`; URLs route back through `<url_prefix>/<key>`."""

    def __init__(self, root: Path, url_prefix: str) -> None:
        self.root = Path(root)
        if not url_prefix.endswith("/"):
            url_prefix = url_prefix + "/"
        self.url_prefix = url_prefix

    def _path(self, key: str) -> Path:
        # keys must be simple filenames, not paths
        if "/" in key or "\\" in key or key in ("", ".", ".."):
            raise ValueError(f"invalid storage key {key!r}")
        return self.root / key

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def url_for(self, key: str) -> str:
        if "/" in key or "\\" in key or key in ("", ".", ".."):
            raise ValueError(f"invalid storage key {key!r}")
        return f"{self.url_prefix}{key}"
