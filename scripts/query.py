import argparse
import sys

from contractiq.config import get_settings
from contractiq.retrieval.naive import get_rag_chain


def print_answer(result: dict) -> None:
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print("-" * 80)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Query the naive RAG pipeline")
    parser.add_argument("question", nargs="*", help="Question to ask")
    args = parser.parse_args()

    settings = get_settings()
    chain, _ = get_rag_chain(settings)

    if args.question:
        print_answer({"question": " ".join(args.question), "answer": chain.invoke(" ".join(args.question))})
        return

    print("Interactive mode. Ctrl+C or empty line to exit.")
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
