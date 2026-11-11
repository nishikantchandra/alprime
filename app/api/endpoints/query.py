"""Grounded RAG question-answering endpoints (JSON + SSE streaming + session management)."""

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import json
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.api.dependencies import get_rag_service, get_session_store
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.schemas.query import (
    Citation,
    QueryRequest,
    QueryResponse,
    SessionTranscriptResponse,
    TranscriptTurnSchema,
)
from app.services.rag_service import RagService
from app.services.session_store import InMemorySessionStore, format_transcript_markdown

logger = get_logger(__name__)
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")

router = APIRouter(
    prefix="/api/v1",
    tags=["RAG Query"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question grounded strictly in uploaded documents",
)
async def query_documents(
    request: QueryRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    """Retrieve matching evidence chunks, rerank them, and generate a cited answer."""
    return await rag_service.answer_question(request)


@router.post(
    "/query/stream",
    summary="Stream grounded answer tokens via Server-Sent Events (SSE)",
)
async def stream_query_documents(
    request: QueryRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> StreamingResponse:
    """Stream citations first, followed by grounded token deltas and completion status."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for item in rag_service.stream_question(request):
                event_name = item["event"]
                payload = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {event_name}\ndata: {payload}\n\n"
        except AppError as exc:
            err_payload = json.dumps({"detail": exc.message, "error_code": exc.error_code})
            yield f"event: error\ndata: {err_payload}\n\n"
        except Exception:
            # Log full details server-side; never leak internals to the client.
            logger.exception("Unexpected failure while streaming an answer")
            err_payload = json.dumps(
                {"detail": "An internal error occurred while generating the answer.", "error_code": "stream_failure"}
            )
            yield f"event: error\ndata: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Clear conversation history for a session",
)
def clear_session_history(
    session_id: str,
    session_store: InMemorySessionStore = Depends(get_session_store),
) -> dict[str, str | bool]:
    """Delete stored turns for `session_id`."""
    existed = session_store.clear_session(session_id)
    if not existed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return {"message": f"Cleared conversation session '{session_id}'.", "cleared": True}


@router.get(
    "/sessions/{session_id}/export",
    summary="Export conversation transcript as Markdown or JSON",
    response_model=None,
)
def export_session_transcript(
    session_id: str,
    format: Literal["markdown", "json"] = Query(
        default="markdown",
        description="Output format: downloadable Markdown (.md) or structured JSON.",
    ),
    session_store: InMemorySessionStore = Depends(get_session_store),
) -> Response:
    """Download the full Q&A transcript with citations for `session_id`."""
    exchanges = session_store.get_transcript(session_id)
    if not exchanges:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcript found for session '{session_id}'.",
        )

    safe_id = _SAFE_FILENAME_RE.sub("-", session_id).strip("-")[:32] or "session"
    if format == "json":
        payload = SessionTranscriptResponse(
            session_id=session_id,
            exported_at=datetime.now(timezone.utc),
            total_turns=len(exchanges),
            turns=[
                TranscriptTurnSchema(
                    turn=item.turn,
                    timestamp=item.timestamp,
                    question=item.question,
                    answer=item.answer,
                    response_type=item.response_type,  # type: ignore[arg-type]
                    is_found=item.is_found,
                    refusal_reason=item.refusal_reason,
                    status_detail=item.status_detail,
                    documents_used=item.documents_used,
                    citations=[Citation(**cit) for cit in item.citations],
                )
                for item in exchanges
            ],
        )
        return JSONResponse(
            content=payload.model_dump(mode="json"),
            headers={
                "Content-Disposition": f'attachment; filename="alprime-transcript-{safe_id}.json"'
            },
        )

    markdown = format_transcript_markdown(session_id, exchanges)
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="alprime-transcript-{safe_id}.md"'
        },
    )
