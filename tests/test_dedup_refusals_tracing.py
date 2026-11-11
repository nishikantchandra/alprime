"""Duplicate-upload detection, natural refusal messages, and per-query trace logging."""

import json
import logging

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.schemas.query import QueryRequest
from app.services.document_parser import ParsedSection
from app.services.prompts import NO_INFO_SENTINEL, NO_INFO_USER_MESSAGE
from app.services.rag_service import RagService, build_refusal_message
from app.services.session_store import InMemorySessionStore
from app.services.text_splitter import chunk_parsed_sections
from tests.conftest import FakeLLMClient, make_memory_vector_store, make_test_settings

_POLICY = b"High-risk client KYC escalation must occur within 1 business day."


def _client(llm: FakeLLMClient | None = None) -> TestClient:
    settings = make_test_settings()
    store = make_memory_vector_store(settings)
    app = create_app(settings=settings, vector_store=store, llm_client=llm or FakeLLMClient())
    return TestClient(app)


def _seed(store, filename: str = "q3_report.pdf") -> None:
    chunks = chunk_parsed_sections(
        [
            ParsedSection(
                text="AlPrime Capital Q3 2026 revenue totaled KWD 1,840,000.",
                page_number=1,
                source_filename=filename,
            )
        ],
        doc_id="doc-1",
        file_type="pdf",
    )
    store.upsert_chunks(chunks, total_pages=1)


# --- Duplicate uploads ----------------------------------------------------------------


def test_reuploading_identical_file_is_not_indexed_twice() -> None:
    with _client() as client:
        first = client.post(
            "/api/v1/documents/upload", files=[("files", ("policy.txt", _POLICY, "text/plain"))]
        )
        assert first.status_code == 201
        original = first.json()["documents"][0]
        assert original["status"] == "indexed"

        # Same bytes under a different name: detected by content hash, not filename.
        again = client.post(
            "/api/v1/documents/upload", files=[("files", ("policy_copy.txt", _POLICY, "text/plain"))]
        )
        assert again.status_code == 200
        body = again.json()
        assert body["duplicates_skipped"] == 1
        assert body["total_chunks_indexed"] == 0
        assert body["documents"][0]["status"] == "already_indexed"
        assert body["documents"][0]["doc_id"] == original["doc_id"]
        assert body["documents"][0]["filename"] == "policy.txt"
        assert "already indexed" in body["message"]

        listing = client.get("/api/v1/documents").json()
        assert listing["total_documents"] == 1
        assert listing["total_chunks"] == original["chunk_count"]


def test_duplicates_inside_one_batch_are_indexed_once() -> None:
    other = b"Advisory retainers are billed monthly in KWD."
    with _client() as client:
        resp = client.post(
            "/api/v1/documents/upload",
            files=[
                ("files", ("a.txt", _POLICY, "text/plain")),
                ("files", ("b.txt", _POLICY, "text/plain")),
                ("files", ("c.txt", other, "text/plain")),
            ],
        )
        assert resp.status_code == 201
        body = resp.json()
        assert [d["filename"] for d in body["documents"]] == ["a.txt", "c.txt"]
        assert body["duplicates_skipped"] == 1
        assert "skipped 1 duplicate" in body["message"]
        assert client.get("/api/v1/documents").json()["total_documents"] == 2


def test_changed_content_is_indexed_as_new_document() -> None:
    with _client() as client:
        client.post("/api/v1/documents/upload", files=[("files", ("p.txt", _POLICY, "text/plain"))])
        edited = _POLICY + b" Exceptions require compliance officer approval."
        resp = client.post("/api/v1/documents/upload", files=[("files", ("p.txt", edited, "text/plain"))])
        assert resp.status_code == 201
        assert resp.json()["documents"][0]["status"] == "indexed"
        assert client.get("/api/v1/documents").json()["total_documents"] == 2


# --- Natural refusals -----------------------------------------------------------------


def test_refusal_messages_lead_clearly_and_name_searched_documents() -> None:
    low = build_refusal_message(
        "low_retrieval_relevance", searched_docs=["q3.pdf", "fees.docx", "policy.txt"]
    )
    assert low.startswith(NO_INFO_USER_MESSAGE)
    assert "q3.pdf, fees.docx and policy.txt" in low
    assert "general knowledge" in low

    many = build_refusal_message("low_retrieval_relevance", searched_docs=[f"d{i}.pdf" for i in range(5)])
    assert "your 5 uploaded documents" in many

    missing = build_refusal_message(
        "insufficient_document_evidence", searched_docs=["q3.pdf", "fees.docx"], evidence_docs=["fees.docx"]
    )
    assert "closest passages I found were in fees.docx" in missing
    assert "q3.pdf" not in missing

    empty = build_refusal_message("no_documents_indexed")
    assert "Upload a PDF, DOCX, or TXT file" in empty


@pytest.mark.asyncio
async def test_off_topic_question_gets_natural_refusal_naming_the_document() -> None:
    settings = make_test_settings(min_vector_score=0.52)
    store = make_memory_vector_store(settings)
    _seed(store)
    llm = FakeLLMClient()
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    resp = await rag.answer_question(QueryRequest(question="Tell everything about how to make a tea"))

    assert resp.is_found is False
    assert resp.refusal_reason == "low_retrieval_relevance"
    assert resp.answer.startswith(NO_INFO_USER_MESSAGE)
    assert "q3_report.pdf" in resp.answer
    assert "relevance checks" in (resp.status_detail or "")
    assert llm.calls == []


@pytest.mark.asyncio
async def test_model_sentinel_refusal_points_to_the_closest_document() -> None:
    store = make_memory_vector_store(make_test_settings(min_vector_score=0.0, min_rerank_score=-100.0))
    _seed(store)
    rag = RagService(
        vector_store=store,
        llm_client=FakeLLMClient(responses=[NO_INFO_SENTINEL]),
        session_store=InMemorySessionStore(),
    )

    resp = await rag.answer_question(QueryRequest(question="What was Q3 2026 revenue in USD?"))

    assert resp.refusal_reason == "insufficient_document_evidence"
    assert "closest passages I found were in q3_report.pdf" in resp.answer
    assert resp.citations == []


# --- Query tracing --------------------------------------------------------------------


def _traces(caplog: pytest.LogCaptureFixture) -> list[dict]:
    prefix = "query_trace "
    return [
        json.loads(r.getMessage()[len(prefix):])
        for r in caplog.records
        if r.getMessage().startswith(prefix)
    ]


@pytest.mark.asyncio
async def test_answer_emits_one_structured_trace(caplog: pytest.LogCaptureFixture) -> None:
    store = make_memory_vector_store(make_test_settings(min_vector_score=0.0, min_rerank_score=-100.0))
    _seed(store)
    rag = RagService(
        vector_store=store,
        llm_client=FakeLLMClient(responses=["Revenue was KWD 1,840,000 [1]."]),
        session_store=InMemorySessionStore(),
    )
    caplog.set_level(logging.INFO, logger="app.trace")

    resp = await rag.answer_question(QueryRequest(question="What was Q3 2026 revenue?"))

    assert resp.is_found is True
    (trace,) = _traces(caplog)
    assert trace["mode"] == "sync"
    assert trace["outcome"] == "grounded"
    assert trace["chunks_retrieved"] >= 1
    assert trace["chunks_cited"] == 1
    assert trace["top_similarity"] is not None
    assert {"scope", "condense", "retrieve_rerank", "llm", "total"} <= set(trace["timings_ms"])
    assert "KWD 1,840,000" not in json.dumps(trace)  # answers / document text are never logged


@pytest.mark.asyncio
async def test_stream_refusal_trace_records_reason(caplog: pytest.LogCaptureFixture) -> None:
    rag = RagService(
        vector_store=make_memory_vector_store(),
        llm_client=FakeLLMClient(),
        session_store=InMemorySessionStore(),
    )
    caplog.set_level(logging.INFO, logger="app.trace")

    events = [e async for e in rag.stream_question(QueryRequest(question="What is the CEO salary?"))]

    assert events[-1]["event"] == "done"
    (trace,) = _traces(caplog)
    assert trace["mode"] == "stream"
    assert trace["outcome"] == "not_found"
    assert trace["refusal_reason"] == "no_documents_indexed"
    assert trace["llm_calls"] == []


@pytest.mark.asyncio
async def test_trace_records_provider_failover_and_token_usage() -> None:
    from types import SimpleNamespace

    from app.core.tracing import report_usage, start_trace
    from app.services.llm_client import RoutingLLMClient
    from tests.test_routing_and_hardening import ScriptedLLM

    class UsageLLM(ScriptedLLM):
        async def generate(self, **kwargs) -> str:
            report_usage(
                SimpleNamespace(
                    prompt_tokens=120,
                    completion_tokens=40,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
                )
            )
            return await super().generate(**kwargs)

    router = RoutingLLMClient(
        [("runware", ScriptedLLM("runware", fail=True)), ("gemini", UsageLLM("gemini", answer="ok"))]
    )
    trace = start_trace(mode="sync", question="q", session_id=None)

    assert await router.generate(system_prompt="s", user_prompt="u") == "ok"

    failed, served = trace.llm_calls
    assert failed["provider"] == "runware" and failed["error"] == "LLMProviderError"
    assert served["provider"] == "gemini" and served["failover_index"] == 1
    assert trace.to_dict()["tokens"] == {
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "reasoning_tokens": 25,
    }


# --- Stalled streaming provider -------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_fails_over_when_provider_sends_no_answer_text() -> None:
    """A provider stuck thinking (connection busy, no answer text) must not hang the UI."""
    import asyncio

    from app.core.tracing import start_trace
    from app.services.llm_client import RoutingLLMClient
    from tests.test_routing_and_hardening import ScriptedLLM

    class StalledLLM(ScriptedLLM):
        closed = False

        async def stream(self, **kwargs):
            try:
                await asyncio.sleep(30)
                yield "never"
            finally:
                StalledLLM.closed = True

    router = RoutingLLMClient(
        [("runware", StalledLLM("runware")), ("gemini", ScriptedLLM("gemini", answer="Revenue was KWD 1,840,000"))],
        first_token_timeout=0.05,
    )
    trace = start_trace(mode="stream", question="q", session_id=None)

    text = "".join([t async for t in router.stream(system_prompt="s", user_prompt="u")])

    assert "KWD 1,840,000" in text
    assert StalledLLM.closed is True
    stalled, served = trace.llm_calls
    assert stalled["provider"] == "runware" and stalled["error"] == "LLMProviderError"
    assert served["provider"] == "gemini" and served["failover_index"] == 1


@pytest.mark.asyncio
async def test_stream_fails_over_when_provider_returns_no_answer_text() -> None:
    """Budget spent entirely on thinking -> empty stream. Must fail over, not read as a refusal."""
    from app.core.tracing import start_trace
    from app.services.llm_client import RoutingLLMClient
    from tests.test_routing_and_hardening import ScriptedLLM

    class EmptyStreamLLM(ScriptedLLM):
        async def stream(self, **kwargs):
            return
            yield  # pragma: no cover - makes this an async generator

    router = RoutingLLMClient(
        [("runware", EmptyStreamLLM("runware")), ("gemini", ScriptedLLM("gemini", answer="Summary [1]"))]
    )
    trace = start_trace(mode="stream", question="q", session_id=None)

    text = "".join([t async for t in router.stream(system_prompt="s", user_prompt="u")])

    assert text.strip() == "Summary [1]"
    empty, served = trace.llm_calls
    assert empty["provider"] == "runware" and empty["error"] == "LLMProviderError"
    assert served["provider"] == "gemini" and served["failover_index"] == 1


def test_runware_completion_budget_leaves_room_for_thinking() -> None:
    from app.config import Settings

    assert Settings.model_fields["runware_max_tokens"].default >= 4096
