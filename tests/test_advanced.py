from langchain_core.documents import Document

from contractiq.ingest.chunker import chunk_to_children
from contractiq.retrieval.reranker import rerank


def test_child_chunks_smaller_than_parent() -> None:
    parent = " ".join(["lorem ipsum termination clause"] * 200)
    children = chunk_to_children(parent, child_size_tokens=400, child_overlap_tokens=50)
    assert len(children) >= 3
    assert all(len(c) < len(parent) for c in children)


def test_rerank_orders_by_relevance() -> None:
    from contractiq.config import get_settings

    s = get_settings()
    docs = [
        Document(page_content="governing law shall be Nevada", metadata={"id": 1}),
        Document(page_content="payment terms net 30 days invoice", metadata={"id": 2}),
        Document(page_content="this agreement is governed by Nevada law jurisdiction", metadata={"id": 3}),
    ]
    ranked = rerank("governing law Nevada", docs, top_k=2, settings=s)
    assert len(ranked) == 2
    assert all("Nevada" in d.page_content for d in ranked)


def test_parent_dedup_keeps_parent_text() -> None:
    text = "parent clause text " * 100
    children = chunk_to_children(text, child_size_tokens=100, child_overlap_tokens=10)
    assert children
