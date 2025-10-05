from __future__ import annotations

from pathlib import Path
from typing import List

from zerocopy.chunker import ChunkSpec, VideoChunker
from zerocopy.config import StorageConfig
from zerocopy.encoder import HashEncoder
from zerocopy.manifest import ManifestStore
from zerocopy.search import TextEncoder, VectorIndex
from zerocopy.services import CompressionService, QueryService


class StubChunker(VideoChunker):
    def __init__(self, chunks: List[ChunkSpec]) -> None:
        self._chunks = chunks
        self.segment_length = 5.0

    def chunk(self, video_path: Path):  # type: ignore[override]
        return self._chunks


class StubEncoder(HashEncoder):
    def __init__(self, vector) -> None:
        super().__init__(embedding_dim=len(vector))
        self._vector = [float(v) for v in vector]

    def encode_chunk(self, chunk_path: Path):  # type: ignore[override]
        return self._vector


class StubTextEncoder(TextEncoder):
    def encode(self, text: str):  # type: ignore[override]
        return [1.0, 0.0, 0.0, 0.0]


def test_compression_and_query(tmp_path: Path) -> None:
    storage = StorageConfig(root_dir=tmp_path, manifest_name="manifest.jsonl", chunk_dir_name="chunks")
    chunk_path = tmp_path / "chunk.mp4"
    chunk_path.write_bytes(b"test")
    chunks = [ChunkSpec(start_ts=0.0, end_ts=2.0, path=chunk_path)]

    chunker = StubChunker(chunks)
    encoder = StubEncoder([1.0, 0.0, 0.0, 0.0])
    index = VectorIndex(embedding_dim=4)
    manifest = ManifestStore(storage)
    service = CompressionService(chunker=chunker, encoder=encoder, manifest=manifest, index=index)

    entries = service.compress(Path("/videos/sample.mp4"), metadata={"label": "car"})
    assert len(entries) == 1
    query_service = QueryService(manifest=manifest, index=index, text_encoder=StubTextEncoder(embedding_dim=4))
    results = query_service.query("car", top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"]["label"] == "car"
