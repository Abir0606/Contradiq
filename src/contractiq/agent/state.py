import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    question: str
    route: str
    documents: list[Document]
    relevant_docs: list[Document]
    retries: int
    generation: str
    citation_issues: list[str]
    fix_attempts: int
    trace: Annotated[list[str], operator.add]


def initial_state(question: str) -> AgentState:
    return {
        "question": question,
        "route": "",
        "documents": [],
        "relevant_docs": [],
        "retries": 0,
        "generation": "",
        "citation_issues": [],
        "fix_attempts": 0,
        "trace": [],
    }
