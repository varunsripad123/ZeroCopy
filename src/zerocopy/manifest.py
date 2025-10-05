"""Manifest storage utilities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .config import StorageConfig
from .logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ManifestEntry:
    """Metadata for a stored latent chunk."""

    chunk_id: str
    source_video: str
    chunk_path: str
    start_ts: float
    end_ts: float
    embedding_path: str
    embedding_dim: int
    metadata: dict


class ManifestStore:
    """JSON lines manifest for persisted chunks."""

    def __init__(self, config: StorageConfig) -> None:
        self._config = config
        self._config.root_dir.mkdir(parents=True, exist_ok=True)
        self._config.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._config.manifest_path.touch(exist_ok=True)
        log.info("manifest.initialized", path=str(self._config.manifest_path))

    @property
    def config(self) -> StorageConfig:
        return self._config

    def append(self, entry: ManifestEntry) -> None:
        log.debug("manifest.append", chunk_id=entry.chunk_id)
        with self._config.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry)) + "\n")

    def load_all(self) -> List[ManifestEntry]:
        entries: List[ManifestEntry] = []
        with self._config.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                entries.append(ManifestEntry(**data))
        log.info("manifest.loaded", count=len(entries))
        return entries

    def find(self, chunk_id: str) -> Optional[ManifestEntry]:
        for entry in self.load_all():
            if entry.chunk_id == chunk_id:
                return entry
        return None

    def write_all(self, entries: Iterable[ManifestEntry]) -> None:
        with self._config.manifest_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(asdict(entry)) + "\n")
        log.info("manifest.write_all")
