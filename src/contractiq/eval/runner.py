import time

from contractiq.config import Settings
from contractiq.eval.metrics import llm_judge, retrieval_hit
from contractiq.retrieval.factory import get_embeddings


def _get_docs_for_mode(question: str, mode: str, settings: Settings, flt=None):
    if mode == "naive":
        from langchain_pinecone import PineconeVectorStore

        vs = PineconeVectorStore(
            index_name=settings.index_name,
            embedding=get_embeddings(settings),
            namespace=settings.namespace_baseline,
            pinecone_api_key=settings.pinecone_api_key,
        )
        retriever = vs.as_retriever(search_kwargs={"k": settings.retrieval_k, **({"filter": flt} if flt else {})})
        return retriever.invoke(question)
    if mode == "hybrid":
        from contractiq.retrieval.hybrid import HybridRetriever

        return HybridRetriever(settings).retrieve(question, filter_dict=flt)
    if mode == "advanced":
        from contractiq.retrieval.advanced import AdvancedRetriever

        orig_r = settings.enable_reranking
        orig_p = settings.enable_parent_retrieval
        settings.enable_reranking = True
        settings.enable_parent_retrieval = True
        try:
            return AdvancedRetriever(settings).retrieve(question, filter_dict=flt)
        finally:
            settings.enable_reranking = orig_r
            settings.enable_parent_retrieval = orig_p
    raise ValueError(mode)


def _generate_answer(question: str, docs: list, settings: Settings) -> str:
    from langchain_core.prompts import ChatPromptTemplate

    from contractiq.retrieval.naive import SYSTEM_PROMPT, format_context

    llm = get_chat_model(settings)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
    context = format_context(docs)
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content


def get_chat_model(settings):
    from contractiq.retrieval.factory import get_chat_model as _g

    return _g(settings)


def evaluate(golden: list[dict], mode: str, settings: Settings, use_llm_judge: bool = True, retrieval_only: bool = False) -> dict:
    results = []
    for item in golden:
        q = item["question"]
        gold = item["gold_answer"]
        flt = {"contract_name": {"$in": [item["contract_file"]]}} if item.get("contract_file") else None
        t0 = time.time()
        docs = _get_docs_for_mode(q, mode, settings, flt=flt)
        t_retrieve = time.time() - t0
        if retrieval_only:
            hit = retrieval_hit(docs, item.get("gold_context", ""), gold)
            results.append(
                {
                    "id": item["id"],
                    "question": q,
                    "gold": gold,
                    "pred": "",
                    "hit": hit,
                    "correct": hit,
                    "latency_retrieve": round(t_retrieve, 3),
                    "latency_gen": 0.0,
                    "answer_type": item["answer_type"],
                }
            )
            continue
        t1 = time.time()
        pred = _generate_answer(q, docs, settings)
        t_gen = time.time() - t1

        hit = retrieval_hit(docs, item.get("gold_context", ""), gold)
        if use_llm_judge:
            correct = llm_judge(q, gold, pred, settings)
        else:
            from contractiq.eval.metrics import contains_answer, yesno_match

            if item["answer_type"] in {"yes", "no"}:
                correct = yesno_match(pred, gold)
            else:
                correct = contains_answer(pred, gold)

        results.append(
            {
                "id": item["id"],
                "question": q,
                "gold": gold,
                "pred": pred[:2000],
                "hit": hit,
                "correct": correct,
                "latency_retrieve": round(t_retrieve, 3),
                "latency_gen": round(t_gen, 3),
                "answer_type": item["answer_type"],
            }
        )
        time.sleep(0.4)

    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    hit_rate = sum(r["hit"] for r in results) / max(len(results), 1)
    by_type = {}
    for at in ["entity", "yes", "no"]:
        subset = [r for r in results if r["answer_type"] == at]
        if subset:
            by_type[at] = round(sum(r["correct"] for r in subset) / len(subset), 3)
    return {
        "mode": mode,
        "n": len(results),
        "accuracy": round(acc, 3),
        "hit_rate": round(hit_rate, 3),
        "by_type": by_type,
        "avg_latency": round(sum(r["latency_retrieve"] + r["latency_gen"] for r in results) / len(results), 3),
        "results": results,
    }
