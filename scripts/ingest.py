import argparse

from contractiq.config import get_settings
from contractiq.ingest.chunker import chunk_contract
from contractiq.ingest.loader import chunk_id, iter_contracts
from contractiq.ingest.vectorstore import (
    chunk_record,
    ensure_index,
    reset_namespace,
    upsert_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CUAD contracts into Pinecone")
    parser.add_argument("--limit", type=int, default=None, help="Max contracts to ingest")
    parser.add_argument("--reset", action="store_true", help="Delete namespace contents first")
    args = parser.parse_args()

    settings = get_settings()
    namespace = settings.namespace_baseline

    ensure_index(settings)
    if args.reset:
        reset_namespace(settings, namespace)

    all_records: list[dict] = []
    contract_count = 0
    for contract in iter_contracts(settings):
        if args.limit is not None and contract_count >= args.limit:
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

    print(f"\ntotal chunks: {len(all_records)} across {contract_count} contracts")
    upserted = upsert_chunks(settings, all_records, namespace=namespace)
    print(f"done: {upserted} vectors in namespace '{namespace}'")


if __name__ == "__main__":
    main()
