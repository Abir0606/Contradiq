from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

from contractiq.config import Settings
from contractiq.retrieval.factory import get_chat_model, get_embeddings

SYSTEM_PROMPT = """You are a legal contract analysis assistant. Answer the user's question \
using ONLY the contract excerpts provided in the context. Rules:
1. Cite the excerpt number in square brackets after every claim, e.g. [2].
2. If the context does not contain the answer, say "I could not find this in the indexed contracts."
3. Quote exact clause language when answering about specific terms.
4. Be precise with dates, party names, and monetary amounts."""


def format_context(docs: list) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{i}] Contract: {meta.get('contract_name', '?')} | Section: {meta.get('section', '?')}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _naive_retriever(settings: Settings, flt: dict | None):
    vectorstore = PineconeVectorStore(
        index_name=settings.index_name,
        embedding=get_embeddings(settings),
        namespace=settings.namespace_baseline,
        pinecone_api_key=settings.pinecone_api_key,
    )
    search_kwargs: dict = {"k": settings.retrieval_k}
    if flt:
        search_kwargs["filter"] = flt
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def _hybrid_retriever_fn(settings: Settings, flt: dict | None):
    from contractiq.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(settings)

    def retrieve(question: str) -> list:
        return retriever.retrieve(question, filter_dict=flt)

    return retrieve


def get_rag_chain(
    settings: Settings,
    mode: str = "naive",
    flt: dict | None = None,
):
    if mode == "advanced":
        from contractiq.retrieval.advanced import AdvancedRetriever

        def retrieve_docs(q: str) -> list:
            return AdvancedRetriever(settings).retrieve(q, filter_dict=flt)

    elif mode == "hybrid":
        retrieve_docs = _hybrid_retriever_fn(settings, flt)
    elif mode == "naive":
        naive_ret = _naive_retriever(settings, flt)
        retrieve_docs = lambda q: naive_ret.invoke(q)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    llm = get_chat_model(settings)

    chain = (
        {
            "context": RunnableLambda(retrieve_docs) | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def answer(chain, question: str) -> dict:
    result = chain.invoke(question)
    return {"question": question, "answer": result}
