"""Document ingestion, listing, and deletion endpoints."""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.api.dependencies import get_app_settings, get_session_store, get_vector_store
from app.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    FileTooLargeError,
    TooManyFilesError,
)
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadedDocResult,
    UploadResponse,
)
from app.services.document_parser import extract_document_sections, sanitize_filename
from app.services.session_store import InMemorySessionStore
from app.services.text_splitter import DocumentChunk, chunk_parsed_sections
from app.services.vector_store import QdrantVectorStore

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)

_READ_CHUNK_BYTES = 1024 * 1024


async def _read_with_limit(upload: UploadFile, max_bytes: int, safe_name: str, max_mb: int) -> bytes:
    """Read an upload in 1 MB pieces and stop as soon as it exceeds the size limit."""
    buffer = bytearray()
    while True:
        piece = await upload.read(_READ_CHUNK_BYTES)
        if not piece:
            break
        buffer.extend(piece)
        if len(buffer) > max_bytes:
            raise FileTooLargeError(safe_name, max_mb)
    return bytes(buffer)


@dataclass(slots=True)
class _PreparedDocument:
    """A parsed and chunked upload, ready to be written to the vector store."""

    result: UploadedDocResult
    chunks: list[DocumentChunk]
    total_pages: int
    content_sha256: str


def _prepare_document(
    raw_bytes: bytes, safe_name: str, content_sha256: str, settings: Settings
) -> _PreparedDocument:
    """Parse and chunk one document (blocking; called via `asyncio.to_thread`)."""
    file_type, sections = extract_document_sections(raw_bytes, safe_name)
    doc_id = str(uuid.uuid4())
    total_pages = max(section.page_number for section in sections)

    chunks = chunk_parsed_sections(
        sections,
        doc_id=doc_id,
        file_type=file_type,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise EmptyDocumentError(f"Document '{safe_name}' produced zero indexable chunks.")

    result = UploadedDocResult(
        doc_id=doc_id,
        filename=safe_name,
        file_type=file_type,
        page_count=total_pages,
        chunk_count=len(chunks),
        char_count=sum(len(section.text) for section in sections),
    )
    return _PreparedDocument(result, chunks, total_pages, content_sha256)


def _already_indexed(existing: DocumentInfo) -> UploadedDocResult:
    return UploadedDocResult(
        doc_id=existing.doc_id,
        filename=existing.filename,
        file_type=existing.file_type,
        page_count=existing.page_count,
        chunk_count=existing.chunk_count,
        status="already_indexed",
    )


async def _index_all(
    prepared_docs: list[_PreparedDocument], vector_store: QdrantVectorStore
) -> tuple[list[UploadedDocResult], int]:
    """Write every prepared document; on any failure remove what this batch already wrote."""
    now = datetime.now(UTC)
    indexed_results: list[UploadedDocResult] = []
    total_chunks = 0
    for doc in prepared_docs:
        try:
            indexed = await asyncio.to_thread(
                vector_store.upsert_chunks,
                doc.chunks,
                total_pages=doc.total_pages,
                uploaded_at=now,
                content_sha256=doc.content_sha256,
            )
        except Exception:
            # Roll back this document (possibly partially written) and the earlier ones.
            for rollback in [*indexed_results, doc.result]:
                try:
                    await asyncio.to_thread(vector_store.delete_document, rollback.doc_id)
                except Exception as cleanup_exc:
                    logger.error("Rollback failed for doc_id=%s: %s", rollback.doc_id, cleanup_exc)
            raise
        total_chunks += indexed
        indexed_results.append(doc.result)
        logger.info(
            "Indexed document '%s' (doc_id=%s, pages=%d, chunks=%d)",
            doc.result.filename,
            doc.result.doc_id,
            doc.result.page_count,
            indexed,
        )
    return indexed_results, total_chunks


def _upload_message(file_count: int, indexed_count: int) -> str:
    skipped = file_count - indexed_count
    if not indexed_count:
        if file_count > 1:
            return f"All {file_count} file(s) were already indexed; nothing new was added."
        return "This file is already indexed; nothing new was added."
    if skipped:
        return (
            f"Indexed {indexed_count} new document(s); "
            f"skipped {skipped} duplicate file(s) that were already indexed."
        )
    return f"Successfully processed and indexed {indexed_count} document(s)."


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index one or more documents (PDF, DOCX, TXT)",
)
async def upload_documents(
    response: Response,
    files: list[UploadFile] = File(..., description="One or more PDF, DOCX, or TXT files."),
    settings: Settings = Depends(get_app_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> UploadResponse:
    """Validate, parse, chunk, embed, and store uploaded documents in Qdrant.

    The batch is all-or-nothing: every file is validated and parsed before anything is
    written, and if indexing fails part-way the documents already indexed are removed.

    Files are de-duplicated by the SHA-256 of their bytes (against the index and within the
    batch): a repeat upload is reported as `already_indexed` with the existing `doc_id`
    instead of indexing the same chunks twice, which would duplicate citations.
    """
    if not files:
        raise EmptyDocumentError("No files were provided in the upload request.")
    if len(files) > settings.max_upload_files:
        raise TooManyFilesError(settings.max_upload_files)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    prepared_docs: list[_PreparedDocument] = []
    duplicates: list[UploadedDocResult] = []
    names_by_hash: dict[str, str] = {}

    for upload in files:
        safe_name = sanitize_filename(upload.filename)
        raw_bytes = await _read_with_limit(upload, max_bytes, safe_name, settings.max_upload_mb)
        digest = hashlib.sha256(raw_bytes).hexdigest()

        existing = await asyncio.to_thread(vector_store.find_document_by_hash, digest)
        if existing is not None:
            logger.info(
                "Skipped duplicate upload '%s' (same content as doc_id=%s '%s')",
                safe_name,
                existing.doc_id,
                existing.filename,
            )
            duplicates.append(_already_indexed(existing))
            continue
        if digest in names_by_hash:
            logger.info("Skipped '%s': same content as '%s' in this batch", safe_name, names_by_hash[digest])
            continue
        names_by_hash[digest] = safe_name

        # CPU-bound parsing/chunking runs in a worker thread so the event loop stays responsive.
        prepared_docs.append(
            await asyncio.to_thread(_prepare_document, raw_bytes, safe_name, digest, settings)
        )

    indexed_results, total_chunks = await _index_all(prepared_docs, vector_store)
    if not indexed_results:
        response.status_code = status.HTTP_200_OK  # nothing was created

    return UploadResponse(
        message=_upload_message(len(files), len(indexed_results)),
        documents=[*indexed_results, *duplicates],
        total_chunks_indexed=total_chunks,
        duplicates_skipped=len(files) - len(indexed_results),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed documents",
)
def list_documents(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> DocumentListResponse:
    """Return all currently indexed documents and vector chunk counts."""
    total_chunks, documents = vector_store.list_documents()
    return DocumentListResponse(
        total_documents=len(documents),
        total_chunks=total_chunks,
        documents=documents,
    )


@router.delete(
    "/reset",
    summary="Clear all documents from the vector store",
)
def reset_knowledge_base(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    session_store: InMemorySessionStore = Depends(get_session_store),
) -> dict[str, str]:
    """Delete the Qdrant collection and clear chat sessions so new uploads start clean."""
    vector_store.reset_collection()
    session_store.clear_all()
    return {"message": "Knowledge base cleared successfully."}


@router.delete(
    "/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a specific document by ID",
)
def delete_document(
    doc_id: str,
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> DeleteDocumentResponse:
    """Remove all indexed chunks associated with `doc_id` from Qdrant."""
    deleted = vector_store.delete_document(doc_id)
    if deleted == 0:
        raise DocumentNotFoundError(doc_id)
    return DeleteDocumentResponse(
        message=f"Deleted document '{doc_id}'.",
        doc_id=doc_id,
        chunks_deleted=deleted,
    )
