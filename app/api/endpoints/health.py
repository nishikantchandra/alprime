"""Operational health and runtime diagnostics endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_app_settings, get_llm_client, get_vector_store
from app.config import Settings
from app.services.llm_client import LLMClient
from app.services.vector_store import QdrantVectorStore

router = APIRouter(tags=["Operational"])


class HealthResponse(BaseModel):
    """Service readiness and runtime dependency snapshot."""

    status: str
    app_name: str
    version: str
    qdrant_mode: str
    embedding_model: str
    reranker_enabled: bool
    llm_provider: str
    llm_configured: bool
    auth_enabled: bool
    indexed_documents: int
    indexed_chunks: int


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & readiness check",
)
def health_check(
    settings: Settings = Depends(get_app_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    llm_client: LLMClient = Depends(get_llm_client),
) -> HealthResponse:
    """Return service status, active RAG configuration, and indexed vector counts."""
    total_chunks, docs = vector_store.list_documents()
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        qdrant_mode="remote" if settings.qdrant_url.strip() else "embedded",
        embedding_model=settings.embedding_model,
        reranker_enabled=settings.enable_reranker,
        llm_provider=llm_client.provider_name,
        llm_configured=llm_client.is_configured,
        auth_enabled=settings.auth_enabled,
        indexed_documents=len(docs),
        indexed_chunks=total_chunks,
    )
