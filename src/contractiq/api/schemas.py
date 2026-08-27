from typing import Literal

from pydantic import BaseModel

Mode = Literal["naive", "hybrid", "advanced"]


class HealthResponse(BaseModel):
    status: str
    index: str
    namespaces: dict


class QueryRequest(BaseModel):
    question: str
    mode: Mode = "hybrid"
    contract_type: str | None = None
    part: str | None = None
    clause: str | None = None
    agent: bool = False


class Source(BaseModel):
    contract_name: str
    section: str
    score: float | None = None
    rerank_score: float | None = None
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    route: str | None = None
    trace: list[str] = []
