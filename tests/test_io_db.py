from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from zerocopy.io import ChunkDatabase, ChunkRecord


def test_chunk_database_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    created_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
    records = [
        ChunkRecord(
            id="chunk-1",
            video_id="video-1",
            t0=0.0,
            t1=2.0,
            path="/tmp/c1.mp4",
            meta={"label": "car"},
            created_at=created_at,
        ),
        ChunkRecord(
            id="chunk-2",
            video_id="video-1",
            t0=2.0,
            t1=4.0,
            path="/tmp/c2.mp4",
            meta=None,
        ),
    ]

    with ChunkDatabase(db_path) as db:
        db.insert_chunks(records)
        first = db.get_chunk("chunk-1")
        assert first is not None
        assert first.meta == {"label": "car"}
        assert first.created_at == created_at

        listing = db.list_chunks_by_video("video-1")
        assert [item.id for item in listing] == ["chunk-1", "chunk-2"]
        assert listing[1].meta == {}

    # reopen database to ensure persistence
    with ChunkDatabase(db_path) as db:
        second = db.get_chunk("chunk-2")
        assert second is not None
        assert second.t0 == pytest.approx(2.0)
        assert second.created_at is not None


def test_chunk_database_noop_on_empty_insert(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    with ChunkDatabase(db_path) as db:
        db.insert_chunks([])
        assert db.list_chunks_by_video("video-x") == []
