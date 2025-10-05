"""API request/response models and model utilities."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .videomae_encoder import VideoMAEEncoder, read_video_frames_rgb


class CompressRequest(BaseModel):
    video_path: str = Field(..., description="Path to a server-accessible video file")
    segment_length: float = Field(2.0, ge=1.0, le=30.0, description="Desired segment length in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata to store with each chunk")


class CompressionResponse(BaseModel):
    video_id: str
    chunk_ids: List[str]


class QueryRequest(BaseModel):
    query: str = Field(..., description="Free-form text prompt")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of hits to return")


class QueryHit(BaseModel):
    chunk_id: str
    score: float
    t0: float
    t1: float
    preview_uri: str
    meta: Dict[str, Any]


class QueryResponse(BaseModel):
    count: int
    hits: List[QueryHit]


class DecodeRequest(BaseModel):
    chunk_id: str


class DecodeResponse(BaseModel):
    chunk_id: str
    uri: str


__all__ = [
    "CompressRequest",
    "CompressionResponse",
    "QueryRequest",
    "QueryResponse",
    "QueryHit",
    "DecodeRequest",
    "DecodeResponse",
    "VideoMAEEncoder",
    "read_video_frames_rgb",
]
