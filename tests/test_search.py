from __future__ import annotations

from zerocopy.search import TextEncoder, VectorIndex


def test_vector_index_query_returns_ordered_results() -> None:
    index = VectorIndex(embedding_dim=4)
    index.add("a", [1.0, 0.0, 0.0, 0.0])
    index.add("b", [0.0, 1.0, 0.0, 0.0])
    query = [0.8, 0.2, 0.0, 0.0]
    results = index.query(query, top_k=2)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_text_encoder_is_deterministic() -> None:
    encoder = TextEncoder(embedding_dim=8)
    vec1 = encoder.encode("hello world")
    vec2 = encoder.encode("hello world")
    assert vec1 == vec2
