"""High level services for compression and search."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List

from .chunker import VideoChunker
from .config import CONFIG
from .encoder import EmbeddingWriter, HashEncoder
from .manifest import ManifestEntry, ManifestStore
from .search import TextEncoder, VectorIndex
from .logging import get_logger

log = get_logger(__name__)


class CompressionService:
    """Handle end-to-end video compression into latent chunks."""

    def __init__(
        self,
        chunker: VideoChunker | None = None,
        encoder: HashEncoder | None = None,
        embedding_writer: EmbeddingWriter | None = None,
        manifest: ManifestStore | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        base_storage = manifest.config if manifest else CONFIG.storage
        self.manifest = manifest or ManifestStore(base_storage)
        self.chunker = chunker or VideoChunker(storage=self.manifest.config)
        self.encoder = encoder or HashEncoder()
        self.embedding_writer = embedding_writer or EmbeddingWriter(self.manifest.config)
        self.manifest.config.root_dir.mkdir(parents=True, exist_ok=True)
        self.manifest.config.upload_dir.mkdir(parents=True, exist_ok=True)
        if index is not None:
            self.index = index
        else:
            self.index = VectorIndex(self.encoder.embedding_dim)

    def compress(self, video_path: Path, metadata: dict | None = None) -> List[ManifestEntry]:
        metadata = metadata or {}
        chunks = self.chunker.chunk(video_path)
        entries: List[ManifestEntry] = []
        for chunk in chunks:
            chunk_id = uuid.uuid4().hex
            embedding = self.encoder.encode_chunk(chunk.path)
            embedding_path = self.embedding_writer.write(chunk_id, embedding)
            entry = ManifestEntry(
                chunk_id=chunk_id,
                source_video=str(video_path),
                chunk_path=str(chunk.path),
                start_ts=chunk.start_ts,
                end_ts=chunk.end_ts,
                embedding_path=str(embedding_path),
                embedding_dim=self.encoder.embedding_dim,
                metadata=metadata,
            )
            self.manifest.append(entry)
            self.index.add(chunk_id, embedding)
            entries.append(entry)
            log.info("compression.chunk_stored", chunk_id=chunk_id)
        return entries


class QueryService:
    """Perform semantic search over stored latents."""

    def __init__(
        self,
        manifest: ManifestStore | None = None,
        index: VectorIndex | None = None,
        text_encoder: TextEncoder | None = None,
    ) -> None:
        storage = manifest.config if manifest else CONFIG.storage
        self.manifest = manifest or ManifestStore(storage)
        if index is not None:
            self.index = index
        else:
            self.index = VectorIndex()
        self.text_encoder = text_encoder or TextEncoder(self.index.embedding_dim)
        if CONFIG.index.rebuild_on_startup:
            self._load_index_from_manifest()

    def _load_index_from_manifest(self) -> None:
        entries = self.manifest.load_all()
        ids: List[str] = []
        vectors: List[List[float]] = []
        for entry in entries:
            embedding_path = Path(entry.embedding_path)
            if not embedding_path.exists():
                continue
            with embedding_path.open("r", encoding="utf-8") as handle:
                vector = json.load(handle)
            if len(vector) != self.index.embedding_dim:
                continue
            ids.append(entry.chunk_id)
            vectors.append(vector)
        self.index.add_bulk(ids, vectors)
        log.info("query.index_rebuilt", count=len(ids))

    def query(self, text: str, top_k: int = 5):
        query_vector = self.text_encoder.encode(text)
        results = self.index.query(query_vector, top_k=top_k)
        enriched = []
        manifest_map = {entry.chunk_id: entry for entry in self.manifest.load_all()}
        for result in results:
            entry = manifest_map.get(result.chunk_id)
            if not entry:
                continue
            enriched.append({
                "chunk_id": result.chunk_id,
                "score": result.score,
                "start_ts": entry.start_ts,
                "end_ts": entry.end_ts,
                "source_video": entry.source_video,
                "chunk_path": entry.chunk_path,
                "metadata": entry.metadata,
            })
        return enriched

    def decode(self, chunk_id: str) -> Path:
        entry = self.manifest.find(chunk_id)
        if not entry:
            raise KeyError(chunk_id)
        return Path(entry.chunk_path)


__all__ = ["CompressionService", "QueryService"]
