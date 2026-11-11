"""Multi-format document parsing for PDF, DOCX (with tables), and TXT files."""

import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFormatError,
)

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt"}
_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """Extracted text unit retaining source filename and logical page number."""

    text: str
    page_number: int
    source_filename: str


def normalize_text(raw: str) -> str:
    """Normalize line endings, strip null bytes, and collapse excessive blank lines."""
    cleaned = raw.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_filename(filename: str | None) -> str:
    """Return a clean basename safe for display and metadata storage."""
    if not filename:
        return "uploaded_document"
    name = Path(filename).name.strip()
    return name or "uploaded_document"


def _parse_pdf(file_bytes: bytes, filename: str) -> list[ParsedSection]:
    """One section per PDF page with text; blank or image-only pages are skipped (no OCR)."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as exc:
        raise DocumentParsingError(f"Invalid or corrupted PDF '{filename}': {exc}") from exc
    except Exception as exc:
        raise DocumentParsingError(f"Failed to open PDF '{filename}': {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise DocumentParsingError(
                f"PDF '{filename}' is encrypted and cannot be parsed without a password."
            ) from exc
        if not decrypted:
            raise DocumentParsingError(f"PDF '{filename}' is password-protected and cannot be parsed.")

    sections: list[ParsedSection] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = normalize_text(page.extract_text() or "")
        except Exception as exc:
            raise DocumentParsingError(f"Error reading page {page_num} of PDF '{filename}': {exc}") from exc
        if page_text:
            sections.append(
                ParsedSection(
                    text=page_text,
                    page_number=page_num,
                    source_filename=filename,
                )
            )

    if not sections:
        raise EmptyDocumentError(
            f"No extractable text found in PDF '{filename}' (it may be blank or scanned image-only)."
        )
    return sections


def _paragraph_has_page_break(paragraph: Paragraph) -> bool:
    """Detect explicit manual page breaks inside a DOCX paragraph XML element."""
    for br in paragraph._element.findall(
        ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
    ):
        br_type = br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type")
        if br_type == "page":
            return True
    return False


def _format_docx_table(table: Table) -> str:
    """Convert a DOCX table to a clean Markdown pipe table for RAG clarity."""
    formatted_rows: list[list[str]] = []
    for row in table.rows:
        cells = [" ".join(cell.text.split()).replace("|", "/") for cell in row.cells]
        if any(cells):
            formatted_rows.append(cells)

    if not formatted_rows:
        return ""

    col_count = max(len(r) for r in formatted_rows)
    padded_rows = [r + [""] * (col_count - len(r)) for r in formatted_rows]

    lines = ["| " + " | ".join(padded_rows[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _parse_docx(file_bytes: bytes, filename: str) -> list[ParsedSection]:
    """Paragraphs and tables in reading order, split into pages at explicit page breaks.

    DOCX has no fixed pagination, so page numbers follow manual breaks in the file; a
    document without breaks is reported as page 1.
    """
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentParsingError(f"Invalid or corrupted DOCX '{filename}': {exc}") from exc

    sections: list[ParsedSection] = []
    current_page = 1
    buffer: list[str] = []

    def flush_page() -> None:
        nonlocal buffer
        joined = normalize_text("\n\n".join(buffer))
        if joined:
            sections.append(
                ParsedSection(
                    text=joined,
                    page_number=current_page,
                    source_filename=filename,
                )
            )
        buffer = []

    # Iterate body elements in natural reading order so headings stay paired with their tables.
    for child in doc.element.body:
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            if text:
                buffer.append(text)
            if _paragraph_has_page_break(paragraph) and buffer:
                flush_page()
                current_page += 1
        elif isinstance(child, CT_Tbl):
            table_md = _format_docx_table(Table(child, doc))
            if table_md:
                buffer.append(table_md)

    flush_page()

    if not sections:
        raise EmptyDocumentError(f"DOCX file '{filename}' contains no extractable text or tables.")
    return sections


def _parse_txt(file_bytes: bytes, filename: str) -> list[ParsedSection]:
    """Decode with the first encoding in `_TEXT_ENCODINGS` that works; the whole file is page 1."""
    if b"\x00" in file_bytes:
        raise DocumentParsingError(f"TXT file '{filename}' appears to contain binary data.")

    decoded: str | None = None
    for encoding in _TEXT_ENCODINGS:
        try:
            decoded = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded is None:
        raise DocumentParsingError(f"Could not decode TXT file '{filename}' with supported encodings.")

    cleaned = normalize_text(decoded)
    if not cleaned:
        raise EmptyDocumentError(f"TXT file '{filename}' is empty or contains only whitespace.")

    return [ParsedSection(text=cleaned, page_number=1, source_filename=filename)]


_PARSERS: dict[str, Callable[[bytes, str], list[ParsedSection]]] = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".txt": _parse_txt,
}


def extract_document_sections(file_bytes: bytes, filename: str | None) -> tuple[str, list[ParsedSection]]:
    """Validate file format/non-emptiness and extract normalized sections with page metadata.

    Returns:
        A tuple of `(file_type_without_dot, list_of_parsed_sections)`.
    """
    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in _PARSERS:
        raise UnsupportedFormatError(safe_name, SUPPORTED_EXTENSIONS)

    if not file_bytes:
        raise EmptyDocumentError(f"Uploaded file '{safe_name}' is 0 bytes.")

    parser = _PARSERS[ext]
    sections = parser(file_bytes, safe_name)
    return ext.lstrip("."), sections
