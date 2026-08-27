import argparse

from contractiq.config import get_settings
from contractiq.ingest.chunker import chunk_contract, chunk_to_children
from contractiq.ingest.loader import chunk_id, iter_contracts
from contractiq.ingest.sparse import BM25SparseEncoder, tokenize
from contractiq.ingest.vectorstore import (
    child_record,
    chunk_record,
    ensure_index,
    reset_namespace,
    upsert_child_chunks,
    upsert_chunks,
    upsert_hybrid_chunks,
)


def _build_parent_records(settings, limit):
    all_records: list[dict] = []
    contract_count = 0
    for contract in iter_contracts(settings):
        if limit is not None and contract_count >= limit:
            break
        chunks = chunk_contract(
            contract["text"],
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        stem = contract["contract_stem"]
        base_metadata = {
            "contract_name": stem + ".pdf",
            "parties": contract["parties"],
            "agreement_date": contract["agreement_date"],
            "contract_type": contract["contract_type"],
            "part": contract["part"],
            "clause_categories": contract["clause_categories"],
            "chunk_prefix": chunk_id(stem, 0).rsplit("-", 1)[0],
        }
        for idx, chunk in enumerate(chunks):
            all_records.append(chunk_record(chunk, base_metadata, idx))
        contract_count += 1
        print(f"chunked [{contract_count}]{stem}: {len(chunks)} chunks")
    return all_records, contract_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CUAD contracts into Pinecone")
    parser.add_argument("--limit", type=int, default=None, help="Max contracts to ingest")
    parser.add_argument("--reset", action="store_true", help="Delete namespace contents first")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Ingest into the hybrid namespace with dense+BM25 sparse vectors",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Ingest child chunks for parent-document retrieval into advanced-v1",
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.advanced:
        namespace = settings.namespace_advanced
        ensure_index(settings)
        if args.reset:
            reset_namespace(settings, namespace)
        parent_records, contract_count = _build_parent_records(settings, args.limit)
        child_records: list[dict] = []
        for rec in parent_records:
            parent_chunk_text = rec["metadata"]["text"]
            children = chunk_to_children(
                parent_chunk_text,
                child_size_tokens=settings.child_chunk_size_tokens,
                child_overlap_tokens=settings.child_chunk_overlap_tokens,
            )
            parent_obj = type("P", (), {"text": parent_chunk_text, "section_breadcrumb": rec["metadata"]["section"]})()
            for child_idx, child_text in enumerate(children):
                child_records.append(
                    child_record(
                        child_text, parent_obj, {k: v for k, v in rec["metadata"].items() if k not in {"text", "parent_text", "parent_id", "child_index"}},
                        rec["metadata"]["chunk_index"],
                        child_idx,
                    )
                )
        print(f"\ntotal parent chunks: {len(parent_records)} across {contract_count} contracts")
        print(f"total child chunks: {len(child_records)}")
        upserted = upsert_child_chunks(settings, child_records, namespace=namespace)
        print(f"done: {upserted} vectors in namespace '{namespace}'")
        return

    namespace = settings.namespace_hybrid if args.hybrid else settings.namespace_baseline

    ensure_index(settings)
    if args.reset:
        reset_namespace(settings, namespace)

    all_records, contract_count = _build_parent_records(settings, args.limit)

    print(f"\ntotal chunks: {len(all_records)} across {contract_count} contracts")

    if args.hybrid:
        print("fitting BM25 sparse encoder on corpus...")
        encoder = BM25SparseEncoder()
        encoder.fit([tokenize(r["text"]) for r in all_records])
        encoder.save(settings.bm25_artifact)
        print(f"vocab size: {encoder.vocab_size}, saved to {settings.bm25_artifact}")
        upserted = upsert_hybrid_chunks(settings, all_records, encoder, namespace=namespace)
    else:
        upserted = upsert_chunks(settings, all_records, namespace=namespace)

    print(f"done: {upserted} vectors in namespace '{namespace}'")


if __name__ == "__main__":
    main()
