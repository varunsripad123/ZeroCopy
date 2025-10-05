from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest

from zerocopy.io.chunker import segment_video


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmpdir = tmp_path_factory.mktemp("video")
    sample_path = tmpdir / "sample.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=30",
        "-t",
        "5",
        "-pix_fmt",
        "yuv420p",
        str(sample_path),
    ]

    subprocess.run(command, check=True)
    return sample_path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_segment_video_creates_expected_chunks(tmp_path: Path, sample_video: Path) -> None:
    out_dir = tmp_path / "chunks"

    manifest = segment_video(str(sample_video), str(out_dir), sec=2)

    assert len(manifest) == 3

    for idx, chunk in enumerate(manifest):
        chunk_path = Path(chunk.path)
        assert chunk_path.exists()
        assert chunk.chunk_id == f"chunk_{idx:04d}"
        assert len(chunk.sha256) == 64
        assert chunk.frames > 0

    # Verify the first two segments are ~2 seconds long and final shorter.
    first_duration = manifest[0].t1 - manifest[0].t0
    second_duration = manifest[1].t1 - manifest[1].t0
    final_duration = manifest[2].t1 - manifest[2].t0

    assert math.isclose(first_duration, 2.0, abs_tol=0.05)
    assert math.isclose(second_duration, 2.0, abs_tol=0.05)
    assert final_duration < 2.0

    # Ensure total duration matches original video within tolerance.
    total_duration = manifest[-1].t1
    assert math.isclose(total_duration, 5.0, abs_tol=0.1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_segment_video_manifest_is_ordered(tmp_path: Path, sample_video: Path) -> None:
    out_dir = tmp_path / "chunks"
    manifest = segment_video(str(sample_video), str(out_dir), sec=2)

    times = [(chunk.t0, chunk.t1) for chunk in manifest]
    assert times == sorted(times)

    hashes = {chunk.sha256 for chunk in manifest}
    assert len(hashes) == len(manifest)
