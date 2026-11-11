"""FastAPI dependency providers backed by application state."""

from fastapi import Request

from app.config import Settings
from app.services.llm_client import LLMClient
from app.services.rag_service import RagService
from app.services.session_store import InMemorySessionStore
from app.services.vector_store import QdrantVectorStore


def get_app_settings(request: Request) -> Settings:
    """Return active Settings attached to app state."""
    return request.app.state.settings


def get_vector_store(request: Request) -> QdrantVectorStore:
    """Return active QdrantVectorStore instance."""
    return request.app.state.vector_store


def get_llm_client(request: Request) -> LLMClient:
    """Return configured LLMClient instance."""
    return request.app.state.llm_client


def get_session_store(request: Request) -> InMemorySessionStore:
    """Return active session store instance."""
    return request.app.state.session_store


def get_rag_service(request: Request) -> RagService:
    """Return active RagService orchestrator."""
    return request.app.state.rag_service
