"""SQLite-backed metadata store for video chunks."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

__all__ = ["ChunkRecord", "ChunkDatabase"]


@dataclass(slots=True)
class ChunkRecord:
    """Representation of a chunk row."""

    id: str
    video_id: str
    t0: float
    t1: float
    path: str
    meta: dict | None = None
    created_at: datetime | None = None


class ChunkDatabase:
    """Lightweight wrapper around SQLite for chunk metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_migrations()

    def close(self) -> None:
        self._conn.close()

    # -- schema -----------------------------------------------------------------
    def _apply_migrations(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA user_version")
        version = cur.fetchone()[0]
        if version < 1:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    t0 REAL NOT NULL,
                    t1 REAL NOT NULL,
                    path TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute("PRAGMA user_version = 1")
            self._conn.commit()

    # -- helpers ----------------------------------------------------------------
    def insert_chunks(self, records: Iterable[ChunkRecord]) -> None:
        payload: List[tuple] = []
        for record in records:
            created = record.created_at or datetime.now(timezone.utc)
            meta = json.dumps(record.meta or {}, ensure_ascii=False)
            payload.append(
                (
                    record.id,
                    record.video_id,
                    float(record.t0),
                    float(record.t1),
                    str(record.path),
                    meta,
                    created.isoformat(),
                )
            )
        if not payload:
            return
        with self._conn:  # implicit transaction
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (id, video_id, t0, t1, path, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_chunks_by_video(self, video_id: str) -> List[ChunkRecord]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM chunks WHERE video_id = ? ORDER BY t0 ASC, id ASC",
            (video_id,),
        )
        rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_chunks(self) -> int:
        """Return the total number of stored chunks."""

        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    # -- conversion -------------------------------------------------------------
    def _row_to_record(self, row: sqlite3.Row) -> ChunkRecord:
        meta_raw = row["meta"]
        meta = json.loads(meta_raw) if meta_raw else {}
        created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
        return ChunkRecord(
            id=row["id"],
            video_id=row["video_id"],
            t0=float(row["t0"]),
            t1=float(row["t1"]),
            path=row["path"],
            meta=meta,
            created_at=created_at,
        )

    # -- context management -----------------------------------------------------
    def __enter__(self) -> "ChunkDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()
