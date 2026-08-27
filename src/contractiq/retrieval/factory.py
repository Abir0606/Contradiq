from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from contractiq.config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.chat_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in .env — get a free key at console.groq.com")
        return ChatGroq(model=settings.chat_model, api_key=settings.groq_api_key, temperature=0)
    if settings.chat_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)
    raise ValueError(f"Unknown chat provider: {settings.chat_provider}")
