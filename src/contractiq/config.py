from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    groq_api_key: str = ""
    pinecone_api_key: str

    index_name: str = "contractiq"
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "dotproduct"

    namespace_baseline: str = "baseline"
    namespace_hybrid: str = "hybrid-v1"
    namespace_advanced: str = "advanced-v1"

    bm25_artifact: Path = ROOT_DIR / "artifacts" / "bm25_encoder.json"

    embedding_provider: str = "huggingface"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    chat_provider: str = "groq"
    chat_model: str = "openai/gpt-oss-120b"

    chunk_size_tokens: int = 1200
    chunk_overlap_tokens: int = 150

    retrieval_k: int = 5
    agent_retrieval_k: int = 6
    max_retrieval_retries: int = 2

    enable_reranking: bool = False
    enable_parent_retrieval: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    parent_chunk_size_tokens: int = 1200
    child_chunk_size_tokens: int = 400
    child_chunk_overlap_tokens: int = 50

    data_dir: Path = ROOT_DIR / "data" / "raw" / "CUAD_v1"

    @property
    def contracts_dir(self) -> Path:
        return self.data_dir / "full_contract_txt"

    @property
    def master_clauses_csv(self) -> Path:
        return self.data_dir / "master_clauses.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()
