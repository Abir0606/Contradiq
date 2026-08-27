import csv
import json
import random
from pathlib import Path

from contractiq.config import get_settings

TEMPLATES = {
    "Agreement Date": "What is the agreement date for the {doc}?",
    "Effective Date": "When does the {doc} become effective?",
    "Expiration Date": "When does the initial term of the {doc} expire?",
    "Parties": "Who are the parties to the {doc}?",
    "Document Name": "What is the document name/title of the file {filename}?",
    "Governing Law": "Which governing law applies to the {doc}?",
    "Renewal Term": "What is the renewal term for the {doc}?",
    "Notice Period To Terminate Renewal": "What notice period is required to terminate renewal of the {doc}?",
}

YESNO_CATEGORIES = [
    "Non-Compete",
    "Exclusivity",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Most Favored Nation",
    "Competitive Restriction Exception",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
]

SYNONYMS = {
    "Non-Compete": "a non-compete restriction",
    "Change Of Control": "a change of control clause",
    "Anti-Assignment": "an anti-assignment restriction",
    "Termination For Convenience": "a termination for convenience right",
    "License Grant": "a license grant",
    "Audit Rights": "audit rights",
    "Cap On Liability": "a cap on liability",
}


def _short_doc(filename: str) -> str:
    stem = filename.removesuffix(".pdf")
    parts = stem.split("_EX-")
    return parts[-1].replace("_", " ") if len(parts) > 1 else stem[:60]


def build_golden(output: Path, n: int = 60, seed: int = 42) -> None:
    random.seed(seed)
    settings = get_settings()
    advanced_count = 150
    all_txt = sorted(settings.contracts_dir.glob("*.txt"))
    allowed_stems = {p.stem for p in all_txt[:advanced_count]}

    with open(settings.master_clauses_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["Filename"].removesuffix(".pdf") in allowed_stems]

    golden: list[dict] = []
    used = set()

    def add_entity():
        for _ in range(500):
            row = random.choice(rows)
            cat = random.choice(list(TEMPLATES.keys()))
            stem = row["Filename"].removesuffix(".pdf")
            if (stem, cat) in used:
                continue
            answer = (row.get(f"{cat}-Answer") or "").strip()
            if not answer or answer.lower() in {"nan", "none"}:
                continue
            doc_short = _short_doc(row["Filename"])
            q = TEMPLATES[cat].format(doc=doc_short, filename=row["Filename"])
            golden.append(
                {
                    "id": f"entity-{len(golden):03d}",
                    "question": q,
                    "gold_answer": answer,
                    "gold_context": (row.get(cat) or "")[:1000],
                    "contract_file": row["Filename"],
                    "category": cat,
                    "answer_type": "entity",
                }
            )
            used.add((stem, cat))
            return True
        return False

    def add_yn(target: str):
        for _ in range(500):
            row = random.choice(rows)
            cat = random.choice(YESNO_CATEGORIES)
            stem = row["Filename"].removesuffix(".pdf")
            if (stem, cat) in used:
                continue
            answer = (row.get(f"{cat}-Answer") or "").strip().lower()
            if answer != target:
                continue
            doc_short = _short_doc(row["Filename"])
            phrase = SYNONYMS.get(cat, f"a {cat.lower()} clause")
            q = f"Does the {doc_short} contain {phrase}?"
            golden.append(
                {
                    "id": f"{target}-{len([g for g in golden if g['answer_type']==target]):03d}",
                    "question": q,
                    "gold_answer": target.capitalize(),
                    "gold_context": (row.get(cat) or "")[:1000] if target == "yes" else "",
                    "contract_file": row["Filename"],
                    "category": cat,
                    "answer_type": target,
                }
            )
            used.add((stem, cat))
            return True
        return False

    for _ in range(20):
        add_entity()
    for _ in range(20):
        add_yn("yes")
    for _ in range(20):
        add_yn("no")

    random.shuffle(golden)
    for i, g in enumerate(golden):
        g["id"] = f"{g['answer_type']}-{i:03d}"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    yes = sum(1 for g in golden if g["answer_type"] == "yes")
    no = sum(1 for g in golden if g["answer_type"] == "no")
    ent = sum(1 for g in golden if g["answer_type"] == "entity")
    print(f"wrote {len(golden)} QAs -> {output} (yes={yes} no={no} entity={ent})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eval/golden.json")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_golden(Path(args.output), n=args.n, seed=args.seed)
