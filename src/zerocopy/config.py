"""Configuration helpers for Zero-Copy AI services."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    return Path(value).expanduser().resolve()


def _env_list(name: str) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(slots=True)
class StorageConfig:
    """Configuration for manifest and chunk storage."""

    root_dir: Path = _env_path("ZEROCOPY_DATA_DIR", "data")
    manifest_name: str = os.getenv("ZEROCOPY_MANIFEST", "manifest.jsonl")
    chunk_dir_name: str = os.getenv("ZEROCOPY_CHUNK_DIR", "chunks")
    upload_dir_name: str = os.getenv("ZEROCOPY_UPLOAD_DIR", "uploads")

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / self.manifest_name

    @property
    def chunk_dir(self) -> Path:
        return self.root_dir / self.chunk_dir_name

    @property
    def upload_dir(self) -> Path:
        return self.root_dir / self.upload_dir_name


@dataclass(slots=True)
class IndexConfig:
    """Configuration for the in-memory index."""

    similarity_metric: str = os.getenv("ZEROCOPY_SIMILARITY", "cosine")
    rebuild_on_startup: bool = os.getenv("ZEROCOPY_REBUILD_INDEX", "false").lower() == "true"


@dataclass(slots=True)
class ModelConfig:
    """Configuration for ML models used by the service."""

    videomae_enabled: bool = os.getenv("ZEROCOPY_VIDEOMAE_ENABLED", "true").lower() == "true"


@dataclass(slots=True)
class ApiConfig:
    host: str = os.getenv("ZEROCOPY_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("ZEROCOPY_API_PORT", "8080"))
    cors_allow_origin: Optional[str] = os.getenv("ZEROCOPY_CORS_ALLOW_ORIGIN")
    cors_allow_origins: List[str] = field(default_factory=lambda: _env_list("ZEROCOPY_CORS_ALLOW_ORIGINS"))


@dataclass(slots=True)
class ZerocopyConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


CONFIG = ZerocopyConfig()
