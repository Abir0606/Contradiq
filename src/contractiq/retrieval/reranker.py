from functools import lru_cache

from contractiq.config import Settings


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def get_reranker(settings):
    return _load_reranker(settings.reranker_model)


def rerank(query: str, docs: list, top_k: int, settings: Settings) -> list:
    if not docs:
        return docs
    reranker = get_reranker(settings)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda pair: -pair[1])
    for doc, score in scored:
        doc.metadata["rerank_score"] = float(score)
    return [doc for doc, _ in scored[:top_k]]
