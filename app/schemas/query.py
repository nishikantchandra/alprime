"""Pydantic schemas for grounded RAG queries, citations, and streaming payloads."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """Exact source provenance and excerpt backing a generated answer."""

    document_name: str = Field(..., description="Filename of the referenced document.")
    doc_id: str = Field(..., description="Unique ID of the referenced document.")
    page: int = Field(..., ge=1, description="Logical page or section number.")
    similarity_score: float = Field(..., description="Vector cosine similarity score from Qdrant.")
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance logit/score when reranker is enabled.",
    )
    snippet: str = Field(..., description="Relevant text excerpt retrieved from the document.")


class QueryRequest(BaseModel):
    """User question request with optional document filtering and session continuity."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to answer strictly from uploaded documents.",
        examples=["What was AlPrime Capital's Q3 2026 consolidated revenue?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of evidence chunks passed to the LLM (defaults to server RERANK_TOP_K).",
    )
    doc_ids: list[str] | None = Field(
        default=None,
        description="Optional list of doc_ids to scope retrieval to specific uploaded documents.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional conversation session identifier for multi-turn follow-up resolution.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_blank_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be blank or whitespace-only.")
        return cleaned


class QueryResponse(BaseModel):
    """Structured grounded response including exact documents and snippets used."""

    answer: str
    is_found: bool = Field(
        ...,
        description="True when grounded evidence was found; False when fallback or conversational.",
    )
    response_type: Literal["grounded", "not_found", "conversational"] = Field(
        default="grounded",
        description="Distinguishes cited answers, missing-evidence refusals, and greetings.",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Specific reason when not_found (no_documents_indexed, selected_documents_missing, low_retrieval_relevance, insufficient_document_evidence).",
    )
    status_detail: str | None = Field(
        default=None,
        description="Human-readable diagnostic explanation of retrieval outcome or conversational mode.",
    )
    documents_used: list[str] = Field(
        default_factory=list,
        description="Deduplicated filenames cited as sources.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ranked source chunks backing the answer.",
    )
    session_id: str | None = Field(
        default=None,
        description="Echoed or newly assigned session ID when conversation history is used.",
    )


class TranscriptTurnSchema(BaseModel):
    """Single exchange recorded inside an exportable session transcript."""

    turn: int
    timestamp: datetime
    question: str
    answer: str
    response_type: Literal["grounded", "not_found", "conversational"] = "grounded"
    is_found: bool = True
    refusal_reason: str | None = None
    status_detail: str | None = None
    documents_used: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class SessionTranscriptResponse(BaseModel):
    """Complete conversation transcript export for a session."""

    session_id: str
    exported_at: datetime
    total_turns: int
    turns: list[TranscriptTurnSchema]
