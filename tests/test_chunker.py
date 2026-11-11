"""Unit tests for recursive text splitting and chunk metadata provenance."""

import pytest

from app.services.document_parser import ParsedSection
from app.services.text_splitter import chunk_parsed_sections, split_text_recursive


def test_split_short_and_empty_text() -> None:
    assert split_text_recursive("   \n\n  ") == []
    assert split_text_recursive("Short sentence.", chunk_size=200, chunk_overlap=50) == [
        "Short sentence."
    ]


def test_split_preserves_overlap_and_size_bounds() -> None:
    sentences = [f"Sentence number {i} contains exact financial metrics." for i in range(1, 20)]
    text = " ".join(sentences)

    chunks = split_text_recursive(text, chunk_size=220, chunk_overlap=60)
    assert len(chunks) > 2
    assert all(len(c) <= 220 for c in chunks)
    # Adjacent chunks should share overlapping content.
    overlap_found = any(chunks[i][-30:] in chunks[i + 1] or chunks[i + 1][:30] in chunks[i] for i in range(len(chunks) - 1))
    assert overlap_found


def test_split_handles_unbroken_token_without_hanging() -> None:
    long_token = "X" * 520
    chunks = split_text_recursive(long_token, chunk_size=200, chunk_overlap=40)
    assert len(chunks) == 3
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks) == long_token


def test_chunk_validation_errors() -> None:
    with pytest.raises(ValueError):
        split_text_recursive("hello", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        split_text_recursive("hello", chunk_size=100, chunk_overlap=100)


def test_chunk_parsed_sections_assigns_deterministic_ids_and_pages() -> None:
    sections = [
        ParsedSection(text="Page one content. " * 15, page_number=1, source_filename="memo.pdf"),
        ParsedSection(text="Page two content.", page_number=2, source_filename="memo.pdf"),
    ]
    chunks = chunk_parsed_sections(
        sections,
        doc_id="doc-123",
        file_type="pdf",
        chunk_size=150,
        chunk_overlap=30,
    )

    assert len(chunks) >= 3
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_id == "doc-123" and c.source_filename == "memo.pdf" for c in chunks)
