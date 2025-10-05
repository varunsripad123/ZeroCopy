"""FastAPI application for Zero-Copy AI."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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

compression_service = CompressionService()
query_service = QueryService(manifest=compression_service.manifest, index=compression_service.index)


def _serialise_entries(entries) -> List[Dict[str, Any]]:
    serialised = []
    for entry in entries:
        serialised.append(
            {
                "chunk_id": entry.chunk_id,
                "start_ts": entry.start_ts,
                "end_ts": entry.end_ts,
                "chunk_path": entry.chunk_path,
                "source_video": entry.source_video,
                "embedding_path": entry.embedding_path,
                "metadata": entry.metadata,
                "chunk_url": f"/chunks/{entry.chunk_id}",
            }
        )
    return serialised


def _serialise_query_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for result in results:
        enriched.append({**result, "chunk_url": f"/chunks/{result['chunk_id']}"})
    return enriched


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

    serialised = _serialise_entries(entries)
    return CompressionResponse(chunk_count=len(entries), entries=serialised)


@app.post("/compress/upload", response_model=CompressionResponse)
async def compress_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    segment_length: float = Form(5.0),
    metadata: str = Form("")
) -> CompressionResponse:
    if segment_length < 0.5 or segment_length > 30.0:
        raise HTTPException(status_code=400, detail="segment_length must be between 0.5 and 30.0 seconds")
    try:
        metadata_dict: Dict[str, Any] = json.loads(metadata) if metadata else {}
        if metadata_dict and not isinstance(metadata_dict, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object") from exc

    filename = file.filename or "upload.mp4"
    suffix = Path(filename).suffix or ".mp4"
    stem = Path(filename).stem or "video"
    stored_path = compression_service.manifest.config.upload_dir / f"{stem}_{uuid.uuid4().hex}{suffix}"

    try:
        with stored_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
    finally:
        background_tasks.add_task(file.close)

    compression_service.chunker.segment_length = segment_length
    try:
        entries = compression_service.compress(stored_path, metadata=metadata_dict)
    except Exception as exc:
        log.exception("api.compress_upload.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Compression failed") from exc

    serialised = _serialise_entries(entries)
    return CompressionResponse(chunk_count=len(entries), entries=serialised)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    results = query_service.query(request.query, top_k=request.top_k)
    serialised = _serialise_query_results(results)
    return QueryResponse(results=serialised)


@app.post("/decode", response_model=DecodeResponse)
def decode(request: DecodeRequest) -> DecodeResponse:
    try:
        chunk_path = query_service.decode(request.chunk_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chunk not found") from exc
    return DecodeResponse(
        chunk_id=request.chunk_id,
        chunk_path=str(chunk_path),
        chunk_url=f"/chunks/{request.chunk_id}",
    )


@app.get("/chunks/{chunk_id}")
def stream_chunk(chunk_id: str):
    try:
        chunk_path = query_service.decode(chunk_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Chunk not found") from exc
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunk file missing")
    return FileResponse(chunk_path, media_type="video/mp4", filename=chunk_path.name)


__all__ = ["app"]
