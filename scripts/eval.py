import argparse
import json
from pathlib import Path

from contractiq.config import get_settings
from contractiq.eval.runner import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ContractIQ pipelines")
    parser.add_argument("--golden", default="data/eval/golden.json")
    parser.add_argument("--modes", nargs="+", default=["naive", "hybrid", "advanced"])
    parser.add_argument("--limit", type=int, default=None, help="Limit QAs for quick run")
    parser.add_argument("--output", default="artifacts/eval_results.json")
    parser.add_argument("--no-llm-judge", action="store_true", help="Use string match instead of LLM judge")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation, report hit_rate only (no LLM tokens)")
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if args.limit:
        golden = golden[: args.limit]
    print(f"loaded {len(golden)} QAs, modes={args.modes}")

    settings = get_settings()
    all_results = {}
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        res = evaluate(golden, mode, settings, use_llm_judge=not args.no_llm_judge, retrieval_only=args.retrieval_only)
        print(f"accuracy={res['accuracy']} hit_rate={res['hit_rate']} by_type={res['by_type']} avg_latency={res['avg_latency']}s")
        all_results[mode] = res

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")

    print("\n| mode | n | accuracy | hit_rate | entity | yes | no | latency |")
    print("|------|---|----------|----------|--------|-----|----|---------|")
    for mode in args.modes:
        r = all_results[mode]
        print(
            f"| {mode:8} | {r['n']:2} | {r['accuracy']:8.3f} | {r['hit_rate']:8.3f} | "
            f"{r['by_type'].get('entity', 0):6.3f} | {r['by_type'].get('yes', 0):4.3f} | {r['by_type'].get('no', 0):3.3f} | {r['avg_latency']:6.3f}s |"
        )


if __name__ == "__main__":
    main()
