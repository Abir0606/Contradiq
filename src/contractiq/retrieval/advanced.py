from contractiq.config import Settings
from contractiq.retrieval.factory import get_embeddings
from contractiq.retrieval.reranker import rerank


class AdvancedRetriever:
    def __init__(self, settings: Settings):
        from pinecone import Pinecone

        self.settings = settings
        self.embeddings = get_embeddings(settings)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.index_name)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_dict: dict | None = None,
    ) -> list:

        final_k = top_k or self.settings.retrieval_k
        do_parent = self.settings.enable_parent_retrieval
        do_rerank = self.settings.enable_reranking

        if do_parent:
            return self._retrieve_parent(query, final_k, filter_dict, do_rerank)
        if do_rerank:
            return self._retrieve_rerank(query, final_k, filter_dict)

        from contractiq.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(self.settings)
        return retriever.retrieve(query, top_k=final_k, filter_dict=filter_dict)

    def _retrieve_rerank(self, query: str, top_k: int, filter_dict: dict | None) -> list:
        from contractiq.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(self.settings)
        candidates = retriever.retrieve(
            query, top_k=self.settings.rerank_candidates, filter_dict=filter_dict
        )
        return rerank(query, candidates, top_k, self.settings)

    def _retrieve_parent(
        self, query: str, top_k: int, filter_dict: dict | None, do_rerank: bool
    ) -> list:
        from langchain_core.documents import Document

        dense = self.embeddings.embed_query(query)
        response = self.index.query(
            vector=dense,
            top_k=self.settings.rerank_candidates,
            namespace=self.settings.namespace_advanced,
            filter=filter_dict,
            include_metadata=True,
        )
        seen: dict[str, Document] = {}
        order: list[str] = []
        for match in response.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            parent_id = metadata.get("parent_id", "")
            parent_text = metadata.pop("parent_text", "") or metadata.get("text", "")
            if parent_id not in seen:
                parent_meta = {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"text", "parent_text", "child_index", "parent_id"}
                }
                parent_meta["parent_id"] = parent_id
                parent_meta["score"] = match.get("score")
                seen[parent_id] = Document(page_content=parent_text, metadata=parent_meta)
                order.append(parent_id)

        parents = [seen[pid] for pid in order]
        if do_rerank:
            return rerank(query, parents, top_k, self.settings)
        return parents[:top_k]
