from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from contractiq.api.schemas import HealthResponse, QueryRequest, QueryResponse, Source
from contractiq.config import get_settings
from contractiq.retrieval.filtering import build_filter

app = FastAPI(title="ContractIQ API", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    from pinecone import Pinecone

    settings = get_settings()
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        stats = pc.Index(settings.index_name).describe_index_stats()
        namespaces = {k: v["vector_count"] for k, v in stats.get("namespaces", {}).items()}
    except Exception:  # noqa: BLE001
        namespaces = {}
    return HealthResponse(status="ok", index=settings.index_name, namespaces=namespaces)


def _run_query(req: QueryRequest) -> QueryResponse:
    from contractiq.agent.graph import build_graph
    from contractiq.agent.state import initial_state

    settings = get_settings()
    if req.mode == "advanced":
        settings.enable_reranking = True
        settings.enable_parent_retrieval = True
    flt = build_filter(req.contract_type, req.part, req.clause)

    if req.agent:
        graph = build_graph(settings, flt=flt)
        state = graph.invoke(initial_state(req.question))
        docs = state.get("relevant_docs") or state.get("documents", [])
        return QueryResponse(
            question=req.question,
            answer=state.get("generation", ""),
            sources=[
                Source(
                    contract_name=d.metadata.get("contract_name", "?"),
                    section=d.metadata.get("section", "?"),
                    score=d.metadata.get("score"),
                    rerank_score=d.metadata.get("rerank_score"),
                    preview=d.page_content[:280].replace("\n", " "),
                )
                for d in docs
            ],
            route=state.get("route"),
            trace=state.get("trace", []),
        )

    from contractiq.retrieval.factory import get_embeddings
    from contractiq.retrieval.naive import get_rag_chain

    settings_tmp = settings
    if req.mode == "advanced":
        settings_tmp.enable_reranking = True
        settings_tmp.enable_parent_retrieval = True

    chain = get_rag_chain(settings_tmp, mode=req.mode, flt=flt)

    docs = []
    if req.mode == "naive":
        from langchain_pinecone import PineconeVectorStore

        vs = PineconeVectorStore(
            index_name=settings.index_name,
            embedding=get_embeddings(settings),
            namespace=settings.namespace_baseline,
            pinecone_api_key=settings.pinecone_api_key,
        )
        docs = vs.as_retriever(search_kwargs={"k": settings.retrieval_k, **({"filter": flt} if flt else {})}).invoke(req.question)
    elif req.mode == "hybrid":
        from contractiq.retrieval.hybrid import HybridRetriever

        docs = HybridRetriever(settings).retrieve(req.question, filter_dict=flt)
    else:
        from contractiq.retrieval.advanced import AdvancedRetriever

        docs = AdvancedRetriever(settings).retrieve(req.question, filter_dict=flt)

    answer = chain.invoke(req.question)
    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=[
            Source(
                contract_name=d.metadata.get("contract_name", "?"),
                section=d.metadata.get("section", "?"),
                score=d.metadata.get("score"),
                rerank_score=d.metadata.get("rerank_score"),
                preview=d.page_content[:280].replace("\n", " "),
            )
            for d in docs
        ],
        trace=[f"mode={req.mode}"],
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return _run_query(req)


@app.post("/agent/query", response_model=QueryResponse)
def agent_query(req: QueryRequest):
    req.agent = True
    return _run_query(req)


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    from contractiq.retrieval.naive import get_rag_chain

    settings = get_settings()
    flt = build_filter(req.contract_type, req.part, req.clause)
    if req.mode == "advanced":
        settings.enable_reranking = True
        settings.enable_parent_retrieval = True
    chain = get_rag_chain(settings, mode=req.mode, flt=flt)

    def gen():
        yield from chain.stream(req.question)

    return StreamingResponse(gen(), media_type="text/plain")
