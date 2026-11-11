"""FastAPI application entrypoint, lifecycle wiring, and Web UI hosting."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path

# Prevent HuggingFace Xet transient Windows socket disconnects & symlink noise.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, setup_logging
from app.services.llm_client import LLMClient, build_llm_client
from app.services.rag_service import RagService
from app.services.session_store import InMemorySessionStore
from app.services.vector_store import QdrantVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
logger = get_logger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    vector_store: QdrantVectorStore | None = None,
    llm_client: LLMClient | None = None,
    session_store: InMemorySessionStore | None = None,
) -> FastAPI:
    """Create configured FastAPI application with cleanly injectable services."""
    resolved_settings = settings or get_settings()
    setup_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        vs = vector_store if vector_store is not None else QdrantVectorStore(resolved_settings)
        llm = llm_client if llm_client is not None else build_llm_client(resolved_settings)
        sessions = session_store if session_store is not None else InMemorySessionStore()

        application.state.vector_store = vs
        application.state.llm_client = llm
        application.state.session_store = sessions
        application.state.rag_service = RagService(
            vector_store=vs,
            llm_client=llm,
            session_store=sessions,
            reasoning_routing=resolved_settings.enable_reasoning_routing,
        )
        logger.info("LLM routing: %s", llm.provider_name)
        try:
            yield
        finally:
            if vector_store is None:
                vs.close()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description=(
            "Production-grade Retrieval-Augmented Generation (RAG) REST API for querying uploaded "
            "PDF, DOCX, and TXT documents with strict anti-hallucination guardrails and citations."
        ),
        lifespan=lifespan,
    )

    # Pre-populate state defensively for callers that instantiate without running lifespan.
    application.state.settings = resolved_settings

    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @application.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "error_code": exc.error_code},
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred.", "error_code": "internal_error"},
        )

    application.include_router(api_router)

    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @application.get("/", include_in_schema=False)
        async def serve_web_ui() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return application


app = create_app()
