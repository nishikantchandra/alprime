"""FastAPI REST endpoint, authentication, error-handling, and SSE integration tests."""

import io

from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from fpdf import FPDF

from app.main import create_app
from app.services.prompts import NO_INFO_SENTINEL, NO_INFO_USER_MESSAGE
from tests.conftest import FakeLLMClient, make_memory_vector_store, make_test_settings


def _sample_pdf_bytes(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, text=text)
    return bytes(pdf.output())


def _sample_docx_bytes(paragraph: str, table_rows: list[list[str]]) -> bytes:
    doc = DocxDocument()
    doc.add_paragraph(paragraph)
    table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
    for r_i, row in enumerate(table_rows):
        for c_i, val in enumerate(row):
            table.cell(r_i, c_i).text = val
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_client(
    *,
    auth_enabled: bool = False,
    api_key: str = "secret-test-key",
    llm_responses: list[str] | None = None,
) -> TestClient:
    settings = make_test_settings(
        auth_enabled=auth_enabled,
        api_key=api_key,
        max_upload_mb=1,
    )
    store = make_memory_vector_store(settings)
    llm = FakeLLMClient(responses=llm_responses or ["Revenue was KWD 1,840,000 [Source: q3.pdf | Page 1]."])
    app = create_app(settings=settings, vector_store=store, llm_client=llm)
    return TestClient(app)


def test_health_and_web_ui_served() -> None:
    with _build_client() as client:
        ui = client.get("/")
        assert ui.status_code == 200
        assert "AlPrime Document Assistant" in ui.text
        assert "Qdrant + FastEmbed" not in ui.text
        assert "exportBtn" in ui.text

        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "healthy"
        assert body["qdrant_mode"] == "embedded"
        assert body["indexed_documents"] == 0


def test_multiformat_upload_list_query_and_delete_lifecycle() -> None:
    pdf_bytes = _sample_pdf_bytes("AlPrime Capital Q3 2026 consolidated revenue was KWD 1,840,000.")
    docx_bytes = _sample_docx_bytes(
        "Advisory Fees",
        [["Tier", "Monthly KWD"], ["CFO Advisory", "1,850"]],
    )
    txt_bytes = b"High-risk client KYC escalation must occur within 1 business day."

    with _build_client() as client:
        upload = client.post(
            "/api/v1/documents/upload",
            files=[
                ("files", ("q3.pdf", pdf_bytes, "application/pdf")),
                (
                    "files",
                    (
                        "fees.docx",
                        docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                ),
                ("files", ("policy.txt", txt_bytes, "text/plain")),
            ],
        )
        assert upload.status_code == 201
        upload_json = upload.json()
        assert len(upload_json["documents"]) == 3
        doc_id = upload_json["documents"][0]["doc_id"]

        listing = client.get("/api/v1/documents")
        assert listing.status_code == 200
        assert listing.json()["total_documents"] == 3

        query = client.post(
            "/api/v1/query",
            json={"question": "What was AlPrime Capital Q3 2026 consolidated revenue?"},
        )
        assert query.status_code == 200
        q_json = query.json()
        assert q_json["is_found"] is True
        assert "q3.pdf" in q_json["documents_used"]
        assert len(q_json["citations"]) >= 1

        deleted = client.delete(f"/api/v1/documents/{doc_id}")
        assert deleted.status_code == 200
        assert deleted.json()["chunks_deleted"] >= 1

        missing_delete = client.delete(f"/api/v1/documents/{doc_id}")
        assert missing_delete.status_code == 404

        reset = client.delete("/api/v1/documents/reset")
        assert reset.status_code == 200
        assert client.get("/api/v1/documents").json()["total_documents"] == 0


def test_upload_and_query_validation_errors() -> None:
    with _build_client() as client:
        bad_ext = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("malware.exe", b"MZ...", "application/octet-stream"))],
        )
        assert bad_ext.status_code == 400
        assert bad_ext.json()["error_code"] == "unsupported_format"

        empty_txt = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("blank.txt", b"   ", "text/plain"))],
        )
        assert empty_txt.status_code == 422
        assert empty_txt.json()["error_code"] == "empty_document"

        oversized = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("big.txt", b"A" * (1024 * 1024 + 10), "text/plain"))],
        )
        assert oversized.status_code == 413

        blank_q = client.post("/api/v1/query", json={"question": "    "})
        assert blank_q.status_code == 422


def test_api_key_authentication_enforced_when_enabled() -> None:
    with _build_client(auth_enabled=True, api_key="alprime-secret") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/documents").status_code == 401
        assert client.get("/api/v1/documents", headers={"X-API-Key": "wrong-key"}).status_code == 401
        assert client.get("/api/v1/documents", headers={"X-API-Key": "alprime-secret"}).status_code == 200


def test_sse_stream_endpoint_and_session_clear() -> None:
    with _build_client(llm_responses=[NO_INFO_SENTINEL]) as client:
        client.post(
            "/api/v1/documents/upload",
            files=[("files", ("notes.txt", b"Kuwait City office opened 2024.", "text/plain"))],
        )
        stream_resp = client.post(
            "/api/v1/query/stream",
            json={"question": "Kuwait City office opened", "session_id": "sess-alpha"},
        )
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        assert NO_INFO_USER_MESSAGE in stream_resp.text

        cleared = client.delete("/api/v1/sessions/sess-alpha")
        assert cleared.status_code == 200
        assert client.delete("/api/v1/sessions/sess-alpha").status_code == 404


def test_session_transcript_export_markdown_and_json() -> None:
    with _build_client(
        llm_responses=[
            "Revenue reached KWD 1,840,000 [1].",
            "High-risk KYC SLA is 1 business day [1].",
        ]
    ) as client:
        # Missing session yields 404 cleanly.
        assert client.get("/api/v1/sessions/missing-sess/export").status_code == 404

        client.post(
            "/api/v1/documents/upload",
            files=[
                ("files", ("q3.txt", b"Q3 2026 consolidated revenue reached KWD 1,840,000.", "text/plain"))
            ],
        )

        # Turn 1: conversational greeting
        g = client.post("/api/v1/query", json={"question": "Hello!", "session_id": "exp-1"})
        assert g.status_code == 200
        assert g.json()["response_type"] == "conversational"

        # Turn 2: grounded answer
        q1 = client.post(
            "/api/v1/query",
            json={"question": "Q3 2026 consolidated revenue", "session_id": "exp-1"},
        )
        assert q1.status_code == 200
        assert q1.json()["is_found"] is True

        # Turn 3: mid-conversation upload + question on new doc (transcript keeps prior turns!)
        client.post(
            "/api/v1/documents/upload",
            files=[
                (
                    "files",
                    ("kyc.txt", b"High-risk client KYC escalation SLA is 1 business day.", "text/plain"),
                )
            ],
        )
        q2 = client.post(
            "/api/v1/query",
            json={"question": "High-risk client KYC escalation SLA", "session_id": "exp-1"},
        )
        assert q2.status_code == 200

        # Export Markdown
        md_resp = client.get("/api/v1/sessions/exp-1/export?format=markdown")
        assert md_resp.status_code == 200
        assert "text/markdown" in md_resp.headers["content-type"]
        assert 'attachment; filename="alprime-transcript-exp-1.md"' in md_resp.headers["content-disposition"]
        assert "Turn 1 — User" in md_resp.text
        assert "Hello!" in md_resp.text
        assert "Revenue reached KWD 1,840,000 [1]." in md_resp.text
        assert "High-risk KYC SLA is 1 business day [1]." in md_resp.text
        assert "**Cited Sources:**" in md_resp.text

        # Export JSON
        json_resp = client.get("/api/v1/sessions/exp-1/export?format=json")
        assert json_resp.status_code == 200
        body = json_resp.json()
        assert body["session_id"] == "exp-1"
        assert body["total_turns"] == 3
        assert [t["response_type"] for t in body["turns"]] == [
            "conversational",
            "grounded",
            "grounded",
        ]
        assert body["turns"][1]["citations"][0]["document_name"] == "q3.txt"
        assert body["turns"][2]["citations"][0]["document_name"] == "kyc.txt"

        # Deleting the session removes exportable transcript too.
        assert client.delete("/api/v1/sessions/exp-1").status_code == 200
        assert client.get("/api/v1/sessions/exp-1/export").status_code == 404
