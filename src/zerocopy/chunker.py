"""Video chunking utilities."""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import StorageConfig
from .logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ChunkSpec:
    start_ts: float
    end_ts: float
    path: Path


class VideoChunker:
    """Chunk a video file into fixed length segments using FFmpeg."""

    def __init__(self, segment_length: float = 5.0, storage: StorageConfig | None = None) -> None:
        if segment_length <= 0:
            raise ValueError("segment_length must be > 0")
        self.segment_length = segment_length
        self.storage = storage or StorageConfig()
        self.storage.chunk_dir.mkdir(parents=True, exist_ok=True)

    def chunk(self, video_path: Path) -> List[ChunkSpec]:
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        duration = self._probe_duration(video_path)
        chunk_specs: List[ChunkSpec] = []

        for index in range(math.ceil(duration / self.segment_length)):
            start = index * self.segment_length
            end = min((index + 1) * self.segment_length, duration)
            chunk_path = self.storage.chunk_dir / f"{video_path.stem}_{index:06d}.mp4"
            self._run_ffmpeg_segment(video_path, chunk_path, start, end - start)
            chunk_specs.append(ChunkSpec(start_ts=start, end_ts=end, path=chunk_path))
        log.info("chunker.completed", chunks=len(chunk_specs))
        return chunk_specs

    @staticmethod
    def _probe_duration(video_path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    @staticmethod
    def _run_ffmpeg_segment(source: Path, target: Path, start: float, duration: float) -> None:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start}",
            "-i",
            str(source),
            "-t",
            f"{duration}",
            "-c",
            "copy",
            str(target),
        ]
        log.debug("chunker.ffmpeg", command=" ".join(command))
        subprocess.run(command, check=True)


__all__ = ["VideoChunker", "ChunkSpec"]
