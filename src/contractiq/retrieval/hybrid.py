from contractiq.config import Settings
from contractiq.ingest.sparse import BM25SparseEncoder
from contractiq.retrieval.factory import get_embeddings


class HybridRetriever:
    def __init__(self, settings: Settings):
        from pinecone import Pinecone

        self.settings = settings
        self.embeddings = get_embeddings(settings)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.index_name)
        self.encoder = BM25SparseEncoder.load(settings.bm25_artifact)

    def retrieve(self, query: str, top_k: int | None = None, filter_dict: dict | None = None) -> list:
        from langchain_core.documents import Document

        k = top_k or self.settings.retrieval_k
        dense = self.embeddings.embed_query(query)
        sparse = self.encoder.encode(query)
        response = self.index.query(
            vector=dense,
            sparse_vector=sparse,
            top_k=k,
            namespace=self.settings.namespace_hybrid,
            filter=filter_dict,
            include_metadata=True,
        )
        docs = []
        for match in response.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            text = metadata.pop("text", "")
            metadata["score"] = match.get("score")
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
