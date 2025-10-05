from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from zerocopy.index import FaissStore


def test_faiss_store_save_and_load(tmp_path) -> None:
    dim = 32
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, dim)).astype("float32")
    ids = [f"chunk-{i}" for i in range(len(vectors))]

    store = FaissStore(dim=dim)
    store.add(ids, vectors)

    query_vec = vectors[0]
    top_before = store.search(query_vec, k=5)

    path = tmp_path / "index.npz"
    store.save(path)

    restored = FaissStore.load(path)
    top_after = restored.search(query_vec, k=5)

    assert [item.chunk_id for item in top_after] == [item.chunk_id for item in top_before]
    assert np.allclose(
        [item.score for item in top_after],
        [item.score for item in top_before],
        atol=1e-5,
    )


def test_faiss_store_validates_dimensions() -> None:
    store = FaissStore(dim=8)
    vecs = np.ones((2, 8), dtype="float32")
    store.add(["a", "b"], vecs)

    with pytest.raises(ValueError):
        store.add(["c"], np.ones((1, 4), dtype="float32"))

    with pytest.raises(ValueError):
        store.search(np.ones((4,), dtype="float32"), k=1)
