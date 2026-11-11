"""Recursive text splitter preserving exact document and page provenance."""

from dataclasses import dataclass
import uuid

from app.services.document_parser import ParsedSection

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Chunked text payload ready for embedding and Qdrant indexing."""

    chunk_id: str
    doc_id: str
    source_filename: str
    file_type: str
    page_number: int
    chunk_index: int
    text: str


def _split_into_atomic_units(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
) -> list[str]:
    """Recursively break text down until every piece is `<= chunk_size`."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    if sep not in text:
        return _split_into_atomic_units(text, remaining_seps, chunk_size)

    raw_parts = text.split(sep)
    # Keep trailing separator attached so concatenating units is lossless.
    parts_with_sep = [p + sep for p in raw_parts[:-1]]
    if raw_parts[-1]:
        parts_with_sep.append(raw_parts[-1])

    units: list[str] = []
    for part in parts_with_sep:
        if not part:
            continue
        if len(part) <= chunk_size:
            units.append(part)
        else:
            units.extend(_split_into_atomic_units(part, remaining_seps, chunk_size))
    return units


def split_text_recursive(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split text into overlapping chunks respecting natural boundaries when possible."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size.")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    units = _split_into_atomic_units(cleaned, separators, chunk_size)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for unit in units:
        if current_parts and (current_len + len(unit) > chunk_size):
            emitted = "".join(current_parts).strip()
            if emitted:
                chunks.append(emitted)

            # Retain up to chunk_overlap tail characters that still leave room for `unit`.
            while current_parts and (
                current_len > chunk_overlap or (current_len + len(unit) > chunk_size)
            ):
                removed = current_parts.pop(0)
                current_len -= len(removed)

        current_parts.append(unit)
        current_len += len(unit)

    tail = "".join(current_parts).strip()
    if tail:
        chunks.append(tail)

    return chunks


def chunk_parsed_sections(
    sections: list[ParsedSection],
    *,
    doc_id: str,
    file_type: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    """Convert parsed sections into indexed `DocumentChunk` records."""
    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for section in sections:
        text_pieces = split_text_recursive(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for piece in text_pieces:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_idx}")),
                    doc_id=doc_id,
                    source_filename=section.source_filename,
                    file_type=file_type,
                    page_number=section.page_number,
                    chunk_index=chunk_idx,
                    text=piece,
                )
            )
            chunk_idx += 1

    return chunks
