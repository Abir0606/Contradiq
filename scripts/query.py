import argparse
import sys

from contractiq.retrieval.filtering import build_filter

from contractiq.config import get_settings
from contractiq.retrieval.naive import get_rag_chain


def print_answer(result: dict) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print("-" * 80)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Query the ContractIQ RAG pipeline")
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument(
        "--mode",
        choices=["naive", "hybrid"],
        default="naive",
        help="Retrieval mode (naive = dense only, hybrid = dense+BM25 sparse)",
    )
    parser.add_argument("--type", dest="contract_type", default=None, help="Filter by contract type")
    parser.add_argument("--part", default=None, help="Filter by CUAD part (Part_I/II/III)")
    parser.add_argument("--clause", dest="clause", default=None, help="Filter by clause category")
    args = parser.parse_args()

    settings = get_settings()
    flt = build_filter(args.contract_type, args.part, args.clause)
    if flt:
        print(f"filter: {flt}")
    chain = get_rag_chain(settings, mode=args.mode, flt=flt)

    if args.question:
        question = " ".join(args.question)
        print_answer({"question": question, "answer": chain.invoke(question)})
        return

    print(f"Interactive mode [{args.mode}]. Ctrl+C or empty line to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            print_answer({"question": question, "answer": chain.invoke(question)})
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
