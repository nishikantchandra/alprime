"""Pydantic schemas for document ingestion, listing, and deletion."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UploadedDocResult(BaseModel):
    """Metadata summary returned for each processed upload."""

    doc_id: str = Field(..., description="Unique document identifier assigned at ingestion.")
    filename: str = Field(..., description="Original sanitized filename.")
    file_type: str = Field(..., description="Normalized file extension (pdf, docx, txt).")
    page_count: int = Field(..., ge=1, description="Number of logical pages or sections parsed.")
    chunk_count: int = Field(..., ge=1, description="Number of text chunks indexed in Qdrant.")
    char_count: int | None = Field(
        default=None,
        ge=1,
        description="Total extracted character count (omitted for already-indexed duplicates).",
    )
    status: Literal["indexed", "already_indexed"] = Field(
        default="indexed",
        description=(
            "`already_indexed` means identical file content (same SHA-256) was uploaded before; "
            "it was not indexed again and `doc_id` points to the existing copy."
        ),
    )


class UploadResponse(BaseModel):
    """Response payload for single or multi-file document upload."""

    message: str
    documents: list[UploadedDocResult]
    total_chunks_indexed: int = Field(..., ge=0)
    duplicates_skipped: int = Field(
        default=0, ge=0, description="Files skipped because identical content was already indexed."
    )


class DocumentInfo(BaseModel):
    """Indexed document metadata entry."""

    doc_id: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    """Listing of all currently indexed documents."""

    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    """Confirmation of document removal from Qdrant."""

    message: str
    doc_id: str
    chunks_deleted: int = Field(..., ge=0)
