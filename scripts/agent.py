import argparse
import sys

from contractiq.agent.graph import build_graph
from contractiq.agent.state import initial_state
from contractiq.config import get_settings
from contractiq.retrieval.filtering import build_filter


def print_result(state: dict) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n--- agent trace ---")
    for step in state["trace"]:
        print(f"  {step}")
    print("\n--- answer ---")
    print(state["generation"])
    docs = state["relevant_docs"] or state["documents"]
    sources = []
    for doc in docs:
        name = doc.metadata.get("contract_name", "?")
        section = doc.metadata.get("section", "?")
        entry = f"{name} | {section}"
        if entry not in sources:
            sources.append(entry)
    if sources and state["route"] == "contract_qa":
        print("\n--- sources ---")
        for src in sources:
            print(f"  {src}")
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="ContractIQ agentic RAG")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--type", dest="contract_type", default=None)
    parser.add_argument("--part", default=None)
    parser.add_argument("--clause", dest="clause", default=None)
    args = parser.parse_args()

    settings = get_settings()
    flt = build_filter(args.contract_type, args.part, args.clause)
    app = build_graph(settings, flt=flt)

    if args.question:
        result = app.invoke(initial_state(" ".join(args.question)))
        print_result(result)
        return

    print(f"Interactive agent mode. Filters: {flt}. Ctrl+C or empty line to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        try:
            result = app.invoke(initial_state(question))
            print_result(result)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
