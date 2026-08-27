from contractiq.config import Settings
from contractiq.ingest.chunker import Chunk


def sanitize_metadata(meta: dict, max_str_len: int = 800) -> dict:
    clean: dict = {}
    for key, value in meta.items():
        if key == "text":
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = value[:max_str_len]
        elif isinstance(value, (int, float, bool)) or (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            clean[key] = value
    return clean


def ensure_index(settings: Settings) -> None:
    import time

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = {index["name"] for index in pc.list_indexes()}
    if settings.index_name in existing:
        desc = pc.describe_index(settings.index_name)
        dim = getattr(desc, "dimension", None) or desc.get("dimension")
        if dim != settings.embedding_dim:
            pc.delete_index(settings.index_name)
            existing.discard(settings.index_name)
            time.sleep(2)
    if settings.index_name not in existing:
        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
        while True:
            status = pc.describe_index(settings.index_name)
            if getattr(status, "ready", False) or status.get("status", {}).get("ready"):
                break
            time.sleep(2)


def reset_namespace(settings: Settings, namespace: str) -> None:
    from pinecone import NotFoundException, Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.index_name)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        pass


def upsert_chunks(
    settings: Settings,
    records: list[dict],
    namespace: str,
    batch_size: int = 96,
) -> int:
    from pinecone import Pinecone

    from contractiq.retrieval.factory import get_embeddings

    embeddings = get_embeddings(settings)
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.index_name)

    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [r["text"] for r in batch]
        vectors_list = embeddings.embed_documents(texts)
        vectors = [
            {
                "id": record["id"],
                "values": values,
                "metadata": sanitize_metadata(record["metadata"]),
            }
            for record, values in zip(batch, vectors_list)
        ]
        index.upsert(vectors=vectors, namespace=namespace)
        total += len(vectors)
        print(f"upserted {total}/{len(records)}")
    return total


def chunk_record(chunk: Chunk, base_metadata: dict, idx: int) -> dict:
    return {
        "id": f"{base_metadata['chunk_prefix']}-{idx:04d}",
        "text": chunk.text,
        "metadata": {
            "text": chunk.text,
            **base_metadata,
            "section": chunk.section_breadcrumb,
            "chunk_index": idx,
        },
    }
