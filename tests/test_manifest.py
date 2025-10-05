from __future__ import annotations

from pathlib import Path

from zerocopy.config import StorageConfig
from zerocopy.manifest import ManifestEntry, ManifestStore


def test_manifest_round_trip(tmp_path: Path) -> None:
    storage = StorageConfig(root_dir=tmp_path, manifest_name="manifest.jsonl", chunk_dir_name="chunks")
    store = ManifestStore(storage)
    entry = ManifestEntry(
        chunk_id="abc123",
        source_video="/videos/sample.mp4",
        chunk_path="/chunks/sample.mp4",
        start_ts=0.0,
        end_ts=2.0,
        embedding_path=str(tmp_path / "abc123.json"),
        embedding_dim=512,
        metadata={"label": "test"},
    )
    store.append(entry)
    loaded = store.load_all()
    assert loaded == [entry]
    assert store.find("abc123") == entry
