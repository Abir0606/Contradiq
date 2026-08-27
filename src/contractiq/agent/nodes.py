from pydantic import BaseModel, Field

from contractiq.config import Settings
from contractiq.retrieval.factory import get_chat_model
from contractiq.retrieval.hybrid import HybridRetriever
from contractiq.retrieval.naive import format_context

ROUTE_SCHEMA = 'Respond with JSON: {"route": "contract_qa" | "general", "reason": "<short>"}'
GRADES_SCHEMA = 'Respond with JSON: {"grades": [true | false, ...]} — one boolean per excerpt, same order'
VERDICT_SCHEMA = (
    'Respond with JSON: {"supported": true|false, "issues": ["<unsupported claim>"], '
    '"revised_answer": "<answer with unsupported claims removed, keep valid citations>"}'
)


class Route(BaseModel):
    route: str
    reason: str = ""


class DocGrades(BaseModel):
    grades: list[bool] = Field(default_factory=list)


class Verdict(BaseModel):
    supported: bool
    issues: list[str] = Field(default_factory=list)
    revised_answer: str = ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def route_question(state, settings: Settings) -> dict:
    llm = get_chat_model(settings).with_structured_output(Route, method="json_mode")
    prompt = (
        "Classify the user question. Use 'contract_qa' if it asks about the contents of "
        "commercial contracts (clauses, parties, dates, terms). Use 'general' for greetings, "
        "small talk, or questions unrelated to contract contents.\n"
        f"{ROUTE_SCHEMA}\nQuestion: {state['question']}"
    )
    result = llm.invoke(prompt)
    return {"route": result.route, "trace": [f"router -> {result.route} ({result.reason})"]}


def respond_general(state, settings: Settings) -> dict:
    llm = get_chat_model(settings)
    prompt = (
        "You are ContractIQ, an assistant that answers questions about a corpus of 510 "
        "indexed commercial contracts. Politely decline to answer anything else and "
        "suggest asking about contract clauses, parties, dates or terms.\n"
        f"User said: {state['question']}"
    )
    return {"generation": llm.invoke(prompt).content}


def retrieve_documents(state, settings: Settings, flt: dict | None) -> dict:
    retriever = HybridRetriever(settings)
    docs = retriever.retrieve(
        state["question"], top_k=settings.agent_retrieval_k, filter_dict=flt
    )
    return {
        "documents": docs,
        "trace": [f"retrieve[{settings.namespace_hybrid}] -> {len(docs)} docs (retry {state['retries']})"],
    }


def grade_documents(state, settings: Settings) -> dict:
    docs = state["documents"]
    if not docs:
        return {"relevant_docs": [], "trace": ["grade -> no docs to grade"]}
    llm = get_chat_model(settings).with_structured_output(DocGrades, method="json_mode")
    excerpts = "\n\n".join(
        f"### Excerpt {i}\n{_clip(d.page_content, 800)}" for i, d in enumerate(docs)
    )
    prompt = (
        "You grade whether contract excerpts help answer a question. An excerpt is "
        "relevant ONLY if it directly addresses the specific thing asked. An excerpt "
        "that merely mentions the same company, product, or topic area as the question "
        "is NOT relevant unless it contains the requested information.\n"
        f"{GRADES_SCHEMA}\nQuestion: {state['question']}\n\n{excerpts}"
    )
    grades = llm.invoke(prompt).grades
    relevant = [doc for doc, keep in zip(docs, grades + [False] * len(docs)) if keep]
    return {
        "relevant_docs": relevant,
        "trace": [f"grade -> {len(relevant)}/{len(docs)} relevant"],
    }


def rewrite_query(state, settings: Settings) -> dict:
    llm = get_chat_model(settings)
    prompt = (
        "Rewrite the question to improve keyword and semantic retrieval over legal "
        "contracts. Keep legal terminology, add synonyms where useful. Return ONLY the "
        f"rewritten question.\nOriginal: {state['question']}"
    )
    rewritten = llm.invoke(prompt).content.strip()
    trace = f"rewrite (retry {state['retries']}): '{rewritten}'"
    return {"question": rewritten, "retries": state["retries"] + 1, "trace": [trace]}


def generate_answer(state, settings: Settings) -> dict:
    llm = get_chat_model(settings)
    docs = state["relevant_docs"] or state["documents"]
    clipped = [
        type(d)(page_content=_clip(d.page_content, 3000), metadata=d.metadata) for d in docs
    ]
    issues = state.get("citation_issues") or []
    rules = (
        "\nYour previous answer had citation problems:\n- "
        + "\n- ".join(issues)
        + "\nFix them: cite only excerpts that genuinely support each claim."
        if issues
        else ""
    )
    system = (
        "You are a legal contract analysis assistant. Answer using ONLY the provided "
        "contract excerpts. Rules:\n"
        "1. Cite the excerpt number in square brackets after every claim, e.g. [2].\n"
        "2. If the excerpts do not contain the answer, say exactly: "
        "'I could not find this in the indexed contracts.'\n"
        "3. Quote exact clause language when answering about specific terms.\n"
        "4. Be precise with dates, party names, and monetary amounts." + rules
    )
    context = format_context(clipped)
    answer = llm.invoke(
        [
            ("system", system),
            ("human", f"Context:\n{context}\n\nQuestion: {state['question']}"),
        ]
    ).content
    return {"generation": answer, "citation_issues": [], "trace": ["generate"]}


def verify_citations(state, settings: Settings) -> dict:
    llm = get_chat_model(settings).with_structured_output(Verdict, method="json_mode")
    docs = state["relevant_docs"] or state["documents"]
    clipped = [
        type(d)(page_content=_clip(d.page_content, 1500), metadata=d.metadata) for d in docs
    ]
    context = format_context(clipped)
    answer_clip = _clip(state["generation"], 2500)
    prompt = (
        "Audit this answer against the numbered excerpts. Every claim followed by a "
        "citation like [2] must be supported by the content of excerpt 2. Claims without "
        "any supporting excerpt must be listed as issues. Keep valid citations in the "
        "revised answer.\n"
        "IMPORTANT: if the answer is a refusal stating the information was not found in "
        "the excerpts, and the excerpts indeed lack that information, mark supported=true "
        "with no issues. Refusals need no citations.\n"
        f"{VERDICT_SCHEMA}\n\nExcerpts:\n{context}\n\nAnswer to audit:"
        f"\n{answer_clip}"
    )
    verdict = llm.invoke(prompt)
    updates = {"fix_attempts": state["fix_attempts"] + 1}
    if verdict.supported:
        updates.update(generation=state["generation"], citation_issues=[], trace=["verify -> OK"])
    else:
        updates.update(
            generation=verdict.revised_answer or state["generation"],
            citation_issues=verdict.issues,
            trace=[f"verify -> {len(verdict.issues)} issue(s): {verdict.issues}"],
        )
    return updates
