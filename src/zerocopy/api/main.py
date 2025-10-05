"""FastAPI surface for the Zero-Copy video pipeline."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..config import CONFIG
from ..index.faiss_store import FaissStore
from ..io.chunker import Chunk, segment_video
from ..io.db import ChunkDatabase, ChunkRecord
from ..models import (
    CompressRequest,
    CompressionResponse,
    DecodeRequest,
    DecodeResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    read_video_frames_rgb,
)
from ..models.videomae_encoder import VideoMAEEncoder
from ..search import TextEncoder
from . import deps

app = FastAPI(title="Zero-Copy AI", version="1.0.0")

cors_origins: List[str]
if CONFIG.api.cors_allow_origins:
    cors_origins = CONFIG.api.cors_allow_origins
elif CONFIG.api.cors_allow_origin:
    cors_origins = [CONFIG.api.cors_allow_origin]
else:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    deps.ensure_storage_directories()


@app.on_event("shutdown")
def _shutdown() -> None:
    store = deps.get_faiss_store()
    deps.persist_faiss_store(store)
    db = deps.get_chunk_database()
    db.close()
    deps.reset_dependencies()


def _validate_source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {resolved}")
    return resolved


def _build_chunk_id(video_id: str, chunk: Chunk) -> str:
    return f"{video_id}_{chunk.chunk_id}"


def _build_chunk_meta(chunk: Chunk, metadata: dict | None) -> dict:
    meta = {
        "segment": {
            "frames": chunk.frames,
            "sha256": chunk.sha256,
        }
    }
    if metadata:
        meta["user"] = metadata
    return meta


@app.post("/compress", response_model=CompressionResponse)
def compress(
    payload: CompressRequest,
    db: ChunkDatabase = Depends(deps.get_chunk_database),
    store: FaissStore = Depends(deps.get_faiss_store),
    encoder: VideoMAEEncoder = Depends(deps.get_videomae_encoder),
) -> CompressionResponse:
    source = _validate_source(Path(payload.video_path))
    segment_seconds = max(1, int(round(payload.segment_length)))

    video_id = uuid.uuid4().hex
    output_dir = CONFIG.storage.chunk_dir / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = segment_video(str(source), str(output_dir), sec=segment_seconds)
    if not chunks:
        return CompressionResponse(video_id=video_id, chunk_ids=[])

    vectors: List[np.ndarray] = []
    chunk_ids: List[str] = []
    records: List[ChunkRecord] = []
    metadata = dict(payload.metadata or {})

    for chunk in chunks:
        frames = read_video_frames_rgb(chunk.path)
        embedding = np.asarray(encoder.encode(frames), dtype=np.float32)
        if embedding.ndim != 1:
            raise HTTPException(status_code=500, detail="Encoder returned invalid embedding shape")
        if embedding.shape[0] != store.dim:
            raise HTTPException(status_code=500, detail="Embedding dimension mismatch with index")
        vectors.append(embedding)

        chunk_id = _build_chunk_id(video_id, chunk)
        chunk_ids.append(chunk_id)
        record = ChunkRecord(
            id=chunk_id,
            video_id=video_id,
            t0=chunk.t0,
            t1=chunk.t1,
            path=chunk.path,
            meta=_build_chunk_meta(chunk, metadata),
        )
        records.append(record)

    db.insert_chunks(records)
    matrix = np.stack(vectors, axis=0)
    store.add(chunk_ids, matrix)
    deps.persist_faiss_store(store)

    return CompressionResponse(video_id=video_id, chunk_ids=chunk_ids)


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    db: ChunkDatabase = Depends(deps.get_chunk_database),
    store: FaissStore = Depends(deps.get_faiss_store),
    text_encoder: TextEncoder = Depends(deps.get_text_encoder),
) -> QueryResponse:
    if len(store) == 0:
        return QueryResponse(count=0, hits=[])

    vector = np.asarray(text_encoder.encode(payload.query), dtype=np.float32)
    results = store.search(vector, payload.top_k)

    hits: List[QueryHit] = []
    for result in results:
        record = db.get_chunk(result.chunk_id)
        if record is None:
            continue
        hits.append(
            QueryHit(
                chunk_id=result.chunk_id,
                score=result.score,
                t0=record.t0,
                t1=record.t1,
                preview_uri=record.path,
                meta=record.meta or {},
            )
        )
    return QueryResponse(count=len(hits), hits=hits)


@app.post("/decode", response_model=DecodeResponse)
def decode(
    payload: DecodeRequest,
    db: ChunkDatabase = Depends(deps.get_chunk_database),
) -> DecodeResponse:
    record = db.get_chunk(payload.chunk_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return DecodeResponse(chunk_id=payload.chunk_id, uri=record.path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics(
    db: ChunkDatabase = Depends(deps.get_chunk_database),
    store: FaissStore = Depends(deps.get_faiss_store),
) -> dict:
    return {
        "chunks": db.count_chunks(),
        "index_size": len(store),
        "index_path": str(deps.get_index_path()),
    }


__all__ = ["app"]
