import os

import httpx
import streamlit as st

API_URL = os.getenv("CONTRACTIQ_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ContractIQ", layout="wide")
st.title("ContractIQ — Agentic RAG for CUAD Contracts")
st.caption("510 commercial contracts · hybrid search · reranking · parent-document retrieval · corrective agent")

with st.sidebar:
    st.header("Pipeline")
    mode = st.selectbox("Retrieval mode", ["naive", "hybrid", "advanced"], index=1)
    use_agent = st.toggle("Agentic mode (router → grade → rewrite → verify)", value=False)
    st.divider()
    st.header("Filters")
    contract_type = st.text_input("contract_type (e.g. Distributor)", value="")
    part = st.selectbox("part", ["", "Part_I", "Part_II", "Part_III"], index=0)
    clause = st.text_input("clause category (e.g. Non-Compete)", value="")
    st.divider()
    if st.button("Health check"):
        try:
            r = httpx.get(f"{API_URL}/health", timeout=10)
            st.json(r.json())
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['contract_name']}** | *{s['section']}* — {s['preview']}")
        if m.get("trace"):
            with st.expander("Agent trace"):
                for t in m["trace"]:
                    st.text(t)

question = st.chat_input("Ask about clauses, parties, dates, termination...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload = {
        "question": question,
        "mode": mode,
        "contract_type": contract_type or None,
        "part": part or None,
        "clause": clause or None,
        "agent": use_agent,
    }
    endpoint = "/agent/query" if use_agent else "/query"
    with st.chat_message("assistant"):
        try:
            r = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=90)
            r.raise_for_status()
            data = r.json()
            st.markdown(data["answer"])
            sources = data.get("sources", [])
            trace = data.get("trace", [])
            if sources:
                with st.expander("Sources", expanded=True):
                    for s in sources:
                        score = f" score={s['score']:.3f}" if s.get("score") else ""
                        st.markdown(f"**{s['contract_name']}** | *{s['section']}*{score}  \n{s['preview']}")
            if trace:
                with st.expander("Agent trace"):
                    for t in trace:
                        st.text(t)
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": sources, "trace": trace}
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"API error: {exc}")
