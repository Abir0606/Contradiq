import csv
import hashlib
from collections.abc import Iterator
from pathlib import Path

from contractiq.config import Settings


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped.lower() in {"none", "nan"}:
        return None
    return stripped


def load_contract_metadata(csv_path: Path) -> dict[str, dict]:
    table: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        answer_columns = [c for c in reader.fieldnames or [] if "answer" in c.lower()]
        category_columns = [
            c.replace("-Answer", "").replace("- Answer", "").strip()
            for c in answer_columns
        ]
        for row in reader:
            filename = row["Filename"].strip()
            stem = filename.removesuffix(".pdf")
            categories = [
                cat
                for cat, col in zip(category_columns, answer_columns)
                if _clean(row.get(col)) is not None
            ]
            table[stem] = {
                "contract_stem": stem,
                "parties": _clean(row.get("Parties-Answer")) or "unknown",
                "agreement_date": _clean(row.get("Agreement Date-Answer")) or "unknown",
                "clause_categories": categories,
            }
    return table


def load_contract_types(pdf_dir: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    if not pdf_dir.exists():
        return mapping
    for part_dir in sorted(pdf_dir.iterdir()):
        if not part_dir.is_dir():
            continue
        for type_dir in sorted(part_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            for pdf_file in type_dir.glob("*.pdf"):
                mapping[pdf_file.stem] = (part_dir.name, type_dir.name)
    return mapping


def iter_contracts(settings: Settings) -> Iterator[dict]:
    metadata_table = load_contract_metadata(settings.master_clauses_csv)
    type_table = load_contract_types(settings.data_dir / "full_contract_pdf")
    txt_files = sorted(settings.contracts_dir.glob("*.txt"))
    matched = 0
    for txt_path in txt_files:
        stem = txt_path.name.removesuffix(".txt")
        meta = metadata_table.get(stem)
        if meta is None:
            meta = {
                "contract_stem": stem,
                "parties": "unknown",
                "agreement_date": "unknown",
                "clause_categories": [],
            }
        part, contract_type = type_table.get(stem, ("unknown", "unknown"))
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        matched += 1
        yield {**meta, "part": part, "contract_type": contract_type, "text": text}


def chunk_id(stem: str, index: int) -> str:
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{index:04d}"
