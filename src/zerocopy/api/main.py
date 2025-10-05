"""FastAPI application for Zero-Copy AI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..config import CONFIG
from ..models import (
    CompressRequest,
    CompressionResponse,
    DecodeRequest,
    DecodeResponse,
    QueryRequest,
    QueryResponse,
)
from ..services import CompressionService, QueryService
from ..logging import get_logger

log = get_logger(__name__)

app = FastAPI(title="Zero-Copy AI", version="0.1.0")

if CONFIG.api.cors_allow_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[CONFIG.api.cors_allow_origin],
        allow_methods=["*"],
        allow_headers=["*"],
    )

compression_service = CompressionService()
query_service = QueryService(manifest=compression_service.manifest, index=compression_service.index)


@app.post("/compress", response_model=CompressionResponse)
def compress(request: CompressRequest) -> CompressionResponse:
    video_path = Path(request.video_path)
    compression_service.chunker.segment_length = request.segment_length
    try:
        entries = compression_service.compress(video_path, metadata=request.metadata)
    except FileNotFoundError as exc:  # pragma: no cover - passthrough
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected error
        log.exception("api.compress.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Compression failed") from exc

    serialised = [
        {
            "chunk_id": entry.chunk_id,
            "start_ts": entry.start_ts,
            "end_ts": entry.end_ts,
            "chunk_path": entry.chunk_path,
            "embedding_path": entry.embedding_path,
            "metadata": entry.metadata,
        }
        for entry in entries
    ]
    return CompressionResponse(chunk_count=len(entries), entries=serialised)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    results = query_service.query(request.query, top_k=request.top_k)
    return QueryResponse(results=results)


@app.post("/decode", response_model=DecodeResponse)
def decode(request: DecodeRequest) -> DecodeResponse:
    try:
        chunk_path = query_service.decode(request.chunk_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chunk not found") from exc
    return DecodeResponse(chunk_id=request.chunk_id, chunk_path=str(chunk_path))


__all__ = ["app"]
