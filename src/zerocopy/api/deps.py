"""Dependency helpers for the Zero-Copy API."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from ..config import CONFIG
from ..index.faiss_store import FaissStore
from ..io.db import ChunkDatabase
from ..models.videomae_encoder import VideoMAEEncoder
from ..search import TextEncoder

_INDEX_FILENAME = "faiss_index.npz"
_DB_FILENAME = "chunks.db"


def _data_root() -> Path:
    return CONFIG.storage.root_dir


def ensure_storage_directories() -> None:
    """Ensure all configured storage directories exist."""

    root = CONFIG.storage.root_dir
    chunk_dir = CONFIG.storage.chunk_dir
    upload_dir = CONFIG.storage.upload_dir
    for directory in (root, chunk_dir, upload_dir):
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_chunk_database() -> ChunkDatabase:
    ensure_storage_directories()
    db_path = _data_root() / _DB_FILENAME
    return ChunkDatabase(db_path)


def get_index_path() -> Path:
    return _data_root() / _INDEX_FILENAME


@lru_cache
def get_videomae_encoder() -> VideoMAEEncoder:
    ensure_storage_directories()
    encoder = VideoMAEEncoder(use_stub=not CONFIG.models.videomae_enabled)
    encoder.load()
    return encoder


@lru_cache
def get_faiss_store() -> FaissStore:
    ensure_storage_directories()
    index_path = get_index_path()
    if index_path.exists():
        return FaissStore.load(index_path)
    if not CONFIG.models.videomae_enabled:
        dim = VideoMAEEncoder.DEFAULT_EMBEDDING_DIM
    else:
        encoder = get_videomae_encoder()
        dim = encoder.embedding_dim
    return FaissStore(dim=dim)


@lru_cache
def get_text_encoder() -> TextEncoder:
    store = get_faiss_store()
    return TextEncoder(embedding_dim=store.dim)


def persist_faiss_store(store: FaissStore) -> None:
    """Persist the FAISS store to disk."""

    index_path = get_index_path()
    store.save(index_path)


def reset_dependencies() -> None:
    """Clear cached dependency singletons (primarily for tests)."""

    for dependency in (get_chunk_database, get_faiss_store, get_videomae_encoder, get_text_encoder):
        cache_clear = getattr(dependency, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


__all__ = [
    "ensure_storage_directories",
    "get_chunk_database",
    "get_faiss_store",
    "get_index_path",
    "get_text_encoder",
    "get_videomae_encoder",
    "persist_faiss_store",
    "reset_dependencies",
]
