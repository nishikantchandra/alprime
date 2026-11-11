"""Unit tests for PDF, DOCX (paragraphs + tables), and TXT document parsing."""

import io

from docx import Document as DocxDocument
from fpdf import FPDF
import pytest

from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from app.services.document_parser import extract_document_sections


def _make_pdf_bytes(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, text=page_text)
    return bytes(pdf.output())


def _make_docx_bytes(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    doc = DocxDocument()
    for text in paragraphs:
        doc.add_paragraph(text)
    if table_rows:
        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r_idx, row in enumerate(table_rows):
            for c_idx, val in enumerate(row):
                table.cell(r_idx, c_idx).text = val
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_parse_multipage_pdf_preserves_page_numbers() -> None:
    pdf_bytes = _make_pdf_bytes(
        [
            "AlPrime Capital Q1 Revenue was 14.2 million KWD.",
            "Operating margin expanded to 22.5 percent in Kuwait.",
        ]
    )
    file_type, sections = extract_document_sections(pdf_bytes, "nested/path/q1_report.pdf")

    assert file_type == "pdf"
    assert len(sections) == 2
    assert sections[0].page_number == 1
    assert sections[0].source_filename == "q1_report.pdf"
    assert "14.2 million KWD" in sections[0].text
    assert sections[1].page_number == 2
    assert "22.5 percent" in sections[1].text


def test_parse_docx_extracts_paragraphs_and_markdown_tables() -> None:
    docx_bytes = _make_docx_bytes(
        paragraphs=["Quarterly Fee Schedule"],
        table_rows=[
            ["Service Tier", "Monthly Fee (KWD)"],
            ["Advisory Retainer", "1,250"],
            ["Audit Readiness", "2,400"],
        ],
    )
    file_type, sections = extract_document_sections(docx_bytes, "fees.docx")

    assert file_type == "docx"
    assert len(sections) == 1
    text = sections[0].text
    assert "Quarterly Fee Schedule" in text
    assert "| Service Tier | Monthly Fee (KWD) |" in text
    assert "| Advisory Retainer | 1,250 |" in text
    assert "| Audit Readiness | 2,400 |" in text


def test_parse_txt_handles_utf8_and_cp1252() -> None:
    utf8_bytes = "\ufeffNet cash flow rose by 18%.\r\nApproved by CFO.".encode("utf-8")
    file_type, sections = extract_document_sections(utf8_bytes, "notes.txt")
    assert file_type == "txt"
    assert sections[0].page_number == 1
    assert sections[0].text == "Net cash flow rose by 18%.\nApproved by CFO."

    cp1252_bytes = "Revenue \u2013 Q3 forecast".encode("cp1252")
    _, cp_sections = extract_document_sections(cp1252_bytes, "cp1252.txt")
    assert "Revenue" in cp_sections[0].text


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFormatError):
        extract_document_sections(b"col1,col2\n1,2", "sheet.csv")


def test_rejects_empty_and_corrupt_files() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_document_sections(b"", "empty.txt")

    with pytest.raises(EmptyDocumentError):
        extract_document_sections(b"   \n\t  ", "whitespace.txt")

    with pytest.raises(DocumentParsingError):
        extract_document_sections(b"not actually a pdf", "broken.pdf")

    with pytest.raises(DocumentParsingError):
        extract_document_sections(b"not actually a docx", "broken.docx")

    with pytest.raises(DocumentParsingError):
        extract_document_sections(b"abc\x00def", "binary.txt")
