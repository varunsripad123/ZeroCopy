"""API request/response models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CompressRequest(BaseModel):
    video_path: str = Field(..., description="Path to the source video on the server")
    segment_length: float = Field(5.0, ge=0.5, le=30.0, description="Segment length in seconds")
    metadata: Dict[str, Any] | None = Field(default=None, description="Arbitrary metadata to persist")


class CompressionResponse(BaseModel):
    chunk_count: int
    entries: List[Dict[str, Any]]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]


class DecodeRequest(BaseModel):
    chunk_id: str


class DecodeResponse(BaseModel):
    chunk_id: str
    chunk_path: str
    chunk_url: str
