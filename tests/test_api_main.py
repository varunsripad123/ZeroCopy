from __future__ import annotations

import importlib
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient


@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROCOPY_DATA_DIR", str(tmp_path))

    import zerocopy.config as config_module

    importlib.reload(config_module)

    import zerocopy.api.deps as deps

    importlib.reload(deps)

    import zerocopy.api.main as api_main

    importlib.reload(api_main)

    deps.reset_dependencies()

    from zerocopy.index.faiss_store import FaissStore

    store = FaissStore(dim=4)

    @lru_cache
    def stub_get_store() -> FaissStore:
        return store

    monkeypatch.setattr(deps, "get_faiss_store", stub_get_store)
    monkeypatch.setattr(deps, "persist_faiss_store", lambda _store: None)

    class _StubEncoder:
        def __init__(self) -> None:
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))

        def load(self) -> None:  # pragma: no cover - compatibility hook
            return None

        def encode(self, frames):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    encoder = _StubEncoder()

    @lru_cache
    def stub_get_encoder():
        return encoder

    monkeypatch.setattr(deps, "get_videomae_encoder", stub_get_encoder)

    class _StubTextEncoder:
        def encode(self, text: str):
            return [1.0, 0.0, 0.0, 0.0]

    @lru_cache
    def stub_get_text_encoder():
        return _StubTextEncoder()

    monkeypatch.setattr(deps, "get_text_encoder", stub_get_text_encoder)

    chunk_path = tmp_path / "chunk.mp4"
    chunk_path.write_bytes(b"chunk-data")

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source-data")

    from zerocopy.io.chunker import Chunk

    fake_chunk = Chunk(
        chunk_id="chunk_0000",
        path=str(chunk_path),
        t0=0.0,
        t1=2.0,
        frames=60,
        sha256="deadbeef",
    )

    monkeypatch.setattr(api_main, "segment_video", lambda *args, **kwargs: [fake_chunk])
    monkeypatch.setattr(
        api_main,
        "read_video_frames_rgb",
        lambda path, num_frames=None: [np.zeros((2, 2, 3), dtype=np.uint8)],
    )

    client = TestClient(api_main.app)
    yield client, source_video
    client.close()


def test_compress_query_decode_flow(api_client):
    client, source_video = api_client

    compress_resp = client.post(
        "/compress",
        json={"video_path": str(source_video), "segment_length": 2, "metadata": {"label": "vehicle"}},
    )
    assert compress_resp.status_code == 200, compress_resp.text
    payload = compress_resp.json()
    assert "video_id" in payload
    assert payload["chunk_ids"]

    chunk_id = payload["chunk_ids"][0]

    query_resp = client.post("/query", json={"query": "vehicle", "top_k": 5})
    assert query_resp.status_code == 200, query_resp.text
    query_payload = query_resp.json()
    assert query_payload["count"] == 1
    hit = query_payload["hits"][0]
    assert hit["chunk_id"] == chunk_id
    assert hit["preview_uri"].endswith("chunk.mp4")
    assert hit["meta"]["user"]["label"] == "vehicle"

    decode_resp = client.post("/decode", json={"chunk_id": chunk_id})
    assert decode_resp.status_code == 200
    assert decode_resp.json()["uri"].endswith("chunk.mp4")

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok"}

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["chunks"] == 1
    assert metrics["index_size"] == 1
