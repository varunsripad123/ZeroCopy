"""Configuration helpers for Zero-Copy AI services."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    return Path(value).expanduser().resolve()


@dataclass(slots=True)
class StorageConfig:
    """Configuration for manifest and chunk storage."""

    root_dir: Path = _env_path("ZEROCOPY_DATA_DIR", "data")
    manifest_name: str = os.getenv("ZEROCOPY_MANIFEST", "manifest.jsonl")
    chunk_dir_name: str = os.getenv("ZEROCOPY_CHUNK_DIR", "chunks")

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / self.manifest_name

    @property
    def chunk_dir(self) -> Path:
        return self.root_dir / self.chunk_dir_name


@dataclass(slots=True)
class IndexConfig:
    """Configuration for the in-memory index."""

    similarity_metric: str = os.getenv("ZEROCOPY_SIMILARITY", "cosine")
    rebuild_on_startup: bool = os.getenv("ZEROCOPY_REBUILD_INDEX", "false").lower() == "true"


@dataclass(slots=True)
class ApiConfig:
    host: str = os.getenv("ZEROCOPY_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("ZEROCOPY_API_PORT", "8080"))
    cors_allow_origin: Optional[str] = os.getenv("ZEROCOPY_CORS_ALLOW_ORIGIN")


@dataclass(slots=True)
class ZerocopyConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


CONFIG = ZerocopyConfig()
