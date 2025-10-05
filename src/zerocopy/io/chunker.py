"""Utilities for segmenting videos into keyframe-aligned MP4 chunks."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
@dataclass(slots=True)
class Chunk:
    """Metadata describing a segmented video chunk."""

    chunk_id: str
    path: str
    t0: float
    t1: float
    frames: int
    sha256: str


def segment_video(input_path: str, out_dir: str, sec: int = 2) -> list[Chunk]:
    """Split ``input_path`` into playable MP4 chunks of ``sec`` seconds.

    Parameters
    ----------
    input_path:
        Path to the source video to segment.
    out_dir:
        Directory that will contain the segmented MP4 files.
    sec:
        Desired segment length in seconds. Defaults to ``2``.

    Returns
    -------
    list[Chunk]
        A manifest describing the generated chunks.
    """

    if sec <= 0:
        raise ValueError("sec must be positive")

    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_pattern = output_dir / "chunk_%04d.mp4"

    _run_ffmpeg_segment(src, segment_pattern, sec)

    chunk_files = sorted(output_dir.glob("chunk_*.mp4"))
    if not chunk_files:
        return []

    manifest: list[Chunk] = []
    current_t = 0.0

    for index, chunk_path in enumerate(chunk_files):
        meta = _probe_chunk(chunk_path)
        duration = meta["duration"]
        frames = meta["frames"]
        sha256 = _hash_file(chunk_path)

        t0 = current_t
        t1 = current_t + duration
        chunk_id = f"chunk_{index:04d}"
        resolved_path = str(chunk_path.resolve())

        manifest.append(
            Chunk(
                chunk_id=chunk_id,
                path=resolved_path,
                t0=t0,
                t1=t1,
                frames=frames,
                sha256=sha256,
            )
        )
        current_t = t1

    return manifest


def _run_ffmpeg_segment(source: Path, target_pattern: Path, sec: int) -> None:
    # Ensure FFmpeg overwrites existing chunk files for deterministic behaviour.
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-x264-params",
        f"keyint={sec*30}:min-keyint={sec*30}:scenecut=0",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{sec})",
        "-movflags",
        "+faststart",
        "-f",
        "segment",
        "-segment_time",
        str(sec),
        "-reset_timestamps",
        "1",
        str(target_pattern),
    ]

    subprocess.run(command, check=True)


def _probe_chunk(chunk_path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(chunk_path),
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)

    stream = info["streams"][0]
    avg_frame_rate = stream.get("avg_frame_rate", "0/1")

    duration_str = info.get("format", {}).get("duration")
    duration = float(duration_str) if duration_str not in (None, "N/A") else None

    nb_frames = stream.get("nb_frames")
    if nb_frames not in (None, "N/A"):
        frames = int(nb_frames)
        if duration is None:
            fps = _fps_from_rate(avg_frame_rate)
            duration = frames / fps if fps else frames / 30
    else:
        fps = _fps_from_rate(avg_frame_rate)
        if duration is None:
            duration = 0.0
        frames = _frames_from_rate(duration, avg_frame_rate)

    if duration is None:
        duration = 0.0

    return {"duration": float(duration), "frames": frames}


def _frames_from_rate(duration: float, rate_expr: str) -> int:
    fps = _fps_from_rate(rate_expr)
    if fps == 0:
        fps = 30
    return int(round(duration * fps))


def _fps_from_rate(rate_expr: str) -> float:
    num, _, den = rate_expr.partition("/")
    try:
        numerator = float(num)
        denominator = float(den) if den else 1.0
    except ValueError:
        return 0.0

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["Chunk", "segment_video"]
