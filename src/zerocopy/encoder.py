"""Latent encoder implementations."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import StorageConfig
from .logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class LatentEmbedding:
    chunk_id: str
    vector: List[float]
    path: Path


class BaseEncoder:
    embedding_dim: int = 512

    def encode_chunk(self, chunk_path: Path) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError


class HashEncoder(BaseEncoder):
    """Deterministic encoder based on SHA256 hashing."""

    def __init__(self, embedding_dim: int = 512) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        self.embedding_dim = embedding_dim

    def encode_chunk(self, chunk_path: Path) -> List[float]:
        hasher = hashlib.sha256()
        with chunk_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.digest()
        repeated = (digest * (self.embedding_dim // len(digest) + 1))[: self.embedding_dim]
        vector = [byte / 255.0 for byte in repeated]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


class EmbeddingWriter:
    """Utility to persist embeddings alongside manifest entries."""

    def __init__(self, storage: StorageConfig) -> None:
        self.storage = storage
        self.storage.root_dir.mkdir(parents=True, exist_ok=True)

    def write(self, chunk_id: str, embedding: List[float]) -> Path:
        path = self.storage.root_dir / f"{chunk_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(embedding, handle)
        log.debug("encoder.embedding_written", path=str(path))
        return path


__all__ = ["LatentEmbedding", "BaseEncoder", "HashEncoder", "EmbeddingWriter"]
