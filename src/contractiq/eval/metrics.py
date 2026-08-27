import re

from contractiq.config import Settings
from contractiq.retrieval.factory import get_chat_model

YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
REFUSAL_RE = re.compile(r"could not find", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def em_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def contains_answer(pred: str, gold: str) -> bool:
    if not gold:
        return False
    return normalize(gold) in normalize(pred)


def yesno_match(pred: str, gold: str) -> bool:
    is_yes = gold.strip().lower() == "yes"
    has_refusal = bool(REFUSAL_RE.search(pred))
    if is_yes:
        return bool(YES_RE.search(pred)) and not has_refusal
    return bool(NO_RE.search(pred)) or has_refusal


def retrieval_hit(docs: list, gold_context: str, gold_answer: str) -> bool:
    if not gold_context and not gold_answer:
        return True
    haystack = " ".join(d.page_content for d in docs).lower()
    needle = (gold_context or gold_answer).lower().strip()
    if len(needle) > 200:
        needle = needle[:200]
    tokens = [t for t in re.findall(r"[a-z0-9]+", needle) if len(t) > 3][:6]
    if not tokens:
        return False
    return sum(1 for t in tokens if t in haystack) >= max(2, len(tokens) // 2)


def llm_judge(question: str, gold: str, pred: str, settings: Settings) -> bool:
    llm = get_chat_model(settings)
    prompt = (
        "You are a strict evaluator. Question and gold answer are ground truth. "
        "Predicted answer is considered correct if it contains the gold answer's key "
        "information, even if phrased differently. For Yes/No questions, a refusal "
        "'could not find' counts as No. Dates like 5/8/14 and May 8, 2014 are equivalent.\n"
        'Respond with JSON: {"correct": true|false, "reason": "<short>"}\n'
        f"Question: {question}\nGold: {gold}\nPredicted: {pred}"
    )
    from pydantic import BaseModel

    class Verdict(BaseModel):
        correct: bool
        reason: str = ""

    judge = llm.with_structured_output(Verdict, method="json_mode")
    try:
        return judge.invoke(prompt).correct
    except Exception:  # noqa: BLE001
        if gold.lower() in {"yes", "no"}:
            return yesno_match(pred, gold)
        return contains_answer(pred, gold)
