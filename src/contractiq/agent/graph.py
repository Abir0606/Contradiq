from langgraph.graph import END, StateGraph

from contractiq.agent import nodes
from contractiq.agent.state import AgentState
from contractiq.config import Settings


def build_graph(settings: Settings, flt: dict | None = None):
    graph = StateGraph(AgentState)

    graph.add_node("router", lambda s: nodes.route_question(s, settings))
    graph.add_node("respond_general", lambda s: nodes.respond_general(s, settings))
    graph.add_node(
        "retrieve", lambda s: nodes.retrieve_documents(s, settings, flt)
    )
    graph.add_node("grade", lambda s: nodes.grade_documents(s, settings))
    graph.add_node("rewrite", lambda s: nodes.rewrite_query(s, settings))
    graph.add_node("generate", lambda s: nodes.generate_answer(s, settings))
    graph.add_node("verify", lambda s: nodes.verify_citations(s, settings))

    graph.set_entry_point("router")

    def after_router(state: AgentState) -> str:
        return "respond_general" if state["route"] == "general" else "retrieve"

    def after_grade(state: AgentState) -> str:
        if state["relevant_docs"]:
            return "generate"
        if state["retries"] < settings.max_retrieval_retries:
            return "rewrite"
        return "generate"

    def after_verify(state: AgentState) -> str:
        if not state["citation_issues"]:
            return END
        if state["fix_attempts"] <= 1:
            return "generate"
        return END

    graph.add_conditional_edges("router", after_router)
    graph.add_edge("respond_general", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify)

    return graph.compile()
