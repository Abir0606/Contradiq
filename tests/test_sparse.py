from contractiq.ingest.sparse import BM25SparseEncoder, tokenize


def test_encode_roundtrip() -> None:
    corpus = [
        "the governing law of this agreement shall be Nevada",
        "termination for convenience requires sixty days notice",
        "license grant is non transferable and perpetual",
    ] * 10
    encoder = BM25SparseEncoder()
    encoder.fit([tokenize(t) for t in corpus])
    sparse = encoder.encode("This agreement shall be governed by Nevada law")
    assert len(sparse["indices"]) == len(sparse["values"]) > 0
    assert sparse["indices"] == sorted(sparse["indices"])
    assert all(v > 0 for v in sparse["values"])


def test_save_load(tmp_path) -> None:
    corpus = ["governing law clause text here", "cap on liability clause"] * 5
    encoder = BM25SparseEncoder()
    encoder.fit([tokenize(t) for t in corpus])
    path = tmp_path / "bm25.json"
    encoder.save(path)
    loaded = BM25SparseEncoder.load(path)
    original = encoder.encode("liability cap")
    restored = loaded.encode("liability cap")
    assert original == restored


def test_unknown_terms_give_empty() -> None:
    encoder = BM25SparseEncoder()
    encoder.fit([tokenize("governing law nevada")])
    sparse = encoder.encode("zzzqqq xyzzyx")
    assert sparse["indices"] == []
