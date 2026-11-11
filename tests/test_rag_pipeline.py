"""Tests for RAG retrieval, document scoping, multi-turn history, and SSE streaming."""

from datetime import datetime, timedelta, timezone
import pytest

from app.core.exceptions import LLMConfigurationError
from app.schemas.query import QueryRequest
from app.services.document_parser import ParsedSection
from app.services.prompts import NO_INFO_SENTINEL, NO_INFO_USER_MESSAGE
from app.services.rag_service import RagService, is_document_overview_query
from app.services.session_store import InMemorySessionStore
from app.services.text_splitter import chunk_parsed_sections
from tests.conftest import FakeCrossEncoder, FakeLLMClient, make_memory_vector_store, make_test_settings


def _seed_store(store) -> None:
    docs = [
        (
            "doc-fin",
            "q3_report.pdf",
            "pdf",
            1,
            "AlPrime Capital Q3 2026 consolidated revenue reached KWD 1,840,000 with 26.0% operating margin.",
        ),
        (
            "doc-fee",
            "fees.docx",
            "docx",
            1,
            "Fractional CFO Advisory retainer fee is 1,850 KWD per month including 80 analyst hours.",
        ),
    ]
    for doc_id, name, ftype, page, text in docs:
        chunks = chunk_parsed_sections(
            [ParsedSection(text=text, page_number=page, source_filename=name)],
            doc_id=doc_id,
            file_type=ftype,
            chunk_size=400,
            chunk_overlap=50,
        )
        store.upsert_chunks(chunks, total_pages=1)


@pytest.mark.asyncio
async def test_grounded_query_returns_answer_and_citations() -> None:
    store = make_memory_vector_store()
    _seed_store(store)
    llm = FakeLLMClient(
        responses=["Consolidated revenue was KWD 1,840,000 [Source: q3_report.pdf | Page 1]."]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    resp = await rag.answer_question(
        QueryRequest(question="What was Q3 2026 consolidated revenue?")
    )

    assert resp.is_found is True
    assert "1,840,000" in resp.answer
    assert "q3_report.pdf" in resp.documents_used
    assert len(resp.citations) >= 1
    assert resp.citations[0].document_name == "q3_report.pdf"
    assert resp.citations[0].page == 1


@pytest.mark.asyncio
async def test_doc_ids_filter_restricts_scope() -> None:
    store = make_memory_vector_store()
    _seed_store(store)
    llm = FakeLLMClient(responses=["Grounded answer."])
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    # Search for CFO Advisory fee while filtering ONLY to q3_report.pdf -> should not leak fees.docx!
    resp = await rag.answer_question(
        QueryRequest(
            question="Fractional CFO Advisory retainer fee monthly analyst hours",
            doc_ids=["doc-fin"],
        )
    )
    assert all(c.doc_id == "doc-fin" for c in resp.citations)
    assert "fees.docx" not in resp.documents_used


@pytest.mark.asyncio
async def test_session_history_condenses_follow_up_query() -> None:
    store = make_memory_vector_store()
    _seed_store(store)
    # First call answers Q3 revenue; second turn sends condense prompt then grounded answer!
    llm = FakeLLMClient(
        responses=[
            "Q3 2026 revenue was KWD 1,840,000 [Source: q3_report.pdf | Page 1].",
            "What was AlPrime Capital's Q3 2026 operating margin?",
            "The Q3 2026 operating margin was 26.0% [Source: q3_report.pdf | Page 1].",
        ]
    )
    sessions = InMemorySessionStore()
    rag = RagService(vector_store=store, llm_client=llm, session_store=sessions)

    await rag.answer_question(
        QueryRequest(question="What was AlPrime Q3 2026 revenue?", session_id="s-1")
    )
    follow_up = await rag.answer_question(
        QueryRequest(question="And what was its operating margin?", session_id="s-1")
    )

    assert follow_up.is_found is True
    assert "26.0%" in follow_up.answer
    assert follow_up.session_id == "s-1"
    # Total LLM calls: turn 1 (1 generate) + turn 2 (condense + answer) = 3 calls.
    assert len(llm.calls) == 3
    assert "Conversation History" in str(llm.calls[1]["user_prompt"])


@pytest.mark.asyncio
async def test_streaming_intercepts_negative_sentinel_without_leaking_token() -> None:
    store = make_memory_vector_store()
    _seed_store(store)
    # Simulate token-by-token streaming of NO_RELEVANT_INFO_FOUND!
    llm = FakeLLMClient(stream_chunks=["NO_RELEVANT_", "INFO_FOUND"])
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    events = [
        ev
        async for ev in rag.stream_question(
            QueryRequest(question="What was Q3 2026 consolidated revenue?")
        )
    ]

    token_texts = [e["data"]["text"] for e in events if e["event"] == "token"]
    assert not any(NO_INFO_SENTINEL in t for t in token_texts)
    done_event = [e for e in events if e["event"] == "done"][-1]
    assert done_event["data"]["is_found"] is False
    assert done_event["data"]["final_answer"].startswith(NO_INFO_USER_MESSAGE)


@pytest.mark.asyncio
async def test_streaming_emits_citations_tokens_and_done_when_grounded() -> None:
    store = make_memory_vector_store()
    _seed_store(store)
    llm = FakeLLMClient(
        stream_chunks=["Retainer fee is ", "1,850 KWD/month ", "[Source: fees.docx | Page 1]."]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    events = [
        ev
        async for ev in rag.stream_question(
            QueryRequest(question="Fractional CFO Advisory retainer fee monthly")
        )
    ]

    event_types = [e["event"] for e in events]
    assert event_types[0] == "status"
    assert "citations" in event_types
    assert "token" in event_types
    assert event_types[-1] == "done"
    done_payload = events[-1]["data"]
    assert done_payload["is_found"] is True
    assert done_payload["duration_ms"] >= 0
    assert "1,850 KWD/month" in done_payload["final_answer"]


@pytest.mark.asyncio
async def test_unconfigured_llm_raises_clean_error_when_chunks_match() -> None:
    store = make_memory_vector_store(make_test_settings())
    _seed_store(store)
    llm = FakeLLMClient(configured=False)
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    with pytest.raises(LLMConfigurationError):
        await rag.answer_question(QueryRequest(question="Q3 2026 consolidated revenue"))


def test_overview_query_classifier_distinguishes_meta_prompts_from_specific_topics() -> None:
    assert is_document_overview_query("now do you know about the selected doc") is True
    assert is_document_overview_query("Summarize the uploaded PDF please") is True
    assert is_document_overview_query("What is this report about?") is True
    assert is_document_overview_query("what is written in this pdf") is True
    assert is_document_overview_query("tell me everything about this document") is True
    assert (
        is_document_overview_query(
            "summarize Assessment_Report.pdf",
            ignored_words={"assessment", "report", "pdf"},
        )
        is True
    )

    assert is_document_overview_query("What does this report say about VAT tax?") is False
    assert is_document_overview_query("How do I deploy Kubernetes on AWS EKS?") is False
    assert is_document_overview_query("What is the CEO salary in USD?") is False


@pytest.mark.asyncio
async def test_session_scope_invalidates_stale_history_when_document_changes() -> None:
    store = make_memory_vector_store(make_test_settings(min_vector_score=0.0, min_rerank_score=-50.0))
    _seed_store(store)
    llm = FakeLLMClient(responses=["Answer one.", "Answer two."])
    sessions = InMemorySessionStore()
    rag = RagService(vector_store=store, llm_client=llm, session_store=sessions)

    await rag.answer_question(
        QueryRequest(question="What was Q3 2026 revenue?", doc_ids=["doc-fin"], session_id="s-swap")
    )
    # Switching doc_ids within same session_id should clear stale history instead of condensing.
    await rag.answer_question(
        QueryRequest(question="What is the monthly retainer?", doc_ids=["doc-fee"], session_id="s-swap")
    )

    assert len(llm.calls) == 2
    assert all("Conversation History" not in str(call["user_prompt"]) for call in llm.calls)


@pytest.mark.asyncio
async def test_lexical_rescue_recovers_terse_keyword_matches_without_allowing_noise() -> None:
    settings = make_test_settings(min_vector_score=0.45, min_rerank_score=-6.0)
    reranker = FakeCrossEncoder(forced_scores=[-7.7, -11.3])
    store = make_memory_vector_store(settings, reranker=reranker)
    _seed_store(store)

    # Note query keyword "retainer" exists in fees.docx; cosine passes >=0.48 with DeterministicEmbedder
    # only if token matches strongly, so test vector_store directly at realistic score bounds.
    hits = store.search_and_rerank(
        "Fractional CFO Advisory retainer fee monthly",
        min_vector_score=0.45,
        min_rerank_score=-6.0,
    )
    assert any(h.document_name == "fees.docx" for h in hits)
    assert all(h.rerank_score is None or h.rerank_score >= -8.8 for h in hits)


@pytest.mark.asyncio
async def test_overview_query_retrieves_lead_chunks_even_when_reranker_scores_low() -> None:
    settings = make_test_settings(min_vector_score=0.95, min_rerank_score=5.0)
    store = make_memory_vector_store(settings, reranker=FakeCrossEncoder(forced_scores=[-11.0]))
    _seed_store(store)
    llm = FakeLLMClient(
        responses=["This document summarizes advisory fees [Source: fees.docx | Page 1]."]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    resp = await rag.answer_question(
        QueryRequest(question="now do you know about the selected doc", doc_ids=["doc-fee"])
    )

    assert resp.is_found is True
    assert resp.documents_used == ["fees.docx"]
    assert resp.citations[0].page == 1
    assert all(c.doc_id == "doc-fee" for c in resp.citations)


@pytest.mark.asyncio
async def test_conversational_greetings_reply_cleanly_without_false_refusal() -> None:
    store = make_memory_vector_store()
    _seed_store(store)
    llm = FakeLLMClient(responses=["NO_RELEVANT_INFO_FOUND"])
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    greet = await rag.answer_question(QueryRequest(question="Hello, how are you?"))
    assert greet.response_type == "conversational"
    assert "q3_report.pdf" in greet.answer
    assert len(llm.calls) == 0

    thanks = await rag.answer_question(QueryRequest(question="Thank you!"))
    assert thanks.response_type == "conversational"
    assert len(llm.calls) == 0

    # Mixed greeting + ungrounded factual trap must STILL route through RAG refusal.
    trap = await rag.answer_question(QueryRequest(question="hello what is the CEO salary in USD?"))
    assert trap.response_type == "not_found"
    assert trap.is_found is False


@pytest.mark.asyncio
async def test_distinct_refusal_reasons_across_empty_kb_gates_and_llm() -> None:
    # 1. Empty KB
    empty_store = make_memory_vector_store()
    rag_empty = RagService(
        vector_store=empty_store,
        llm_client=FakeLLMClient(),
        session_store=InMemorySessionStore(),
    )
    r_empty = await rag_empty.answer_question(QueryRequest(question="What is revenue?"))
    assert r_empty.refusal_reason == "no_documents_indexed"

    # 2. Retrieval gates block noise
    seeded_store = make_memory_vector_store(make_test_settings(min_vector_score=0.95))
    _seed_store(seeded_store)
    rag_gated = RagService(
        vector_store=seeded_store,
        llm_client=FakeLLMClient(),
        session_store=InMemorySessionStore(),
    )
    r_gated = await rag_gated.answer_question(QueryRequest(question="How to bake cookies?"))
    assert r_gated.refusal_reason == "low_retrieval_relevance"

    # 3. Retrieved chunks lack evidence -> LLM returns sentinel
    permissive_store = make_memory_vector_store(
        make_test_settings(min_vector_score=0.0, min_rerank_score=-50.0)
    )
    _seed_store(permissive_store)
    rag_llm_refuse = RagService(
        vector_store=permissive_store,
        llm_client=FakeLLMClient(responses=[NO_INFO_SENTINEL]),
        session_store=InMemorySessionStore(),
    )
    r_insuff = await rag_llm_refuse.answer_question(QueryRequest(question="Q3 2026 revenue?"))
    assert r_insuff.refusal_reason == "insufficient_document_evidence"


@pytest.mark.asyncio
async def test_mid_conversation_new_upload_resets_stale_scope_and_answers_new_doc() -> None:
    store = make_memory_vector_store(make_test_settings(min_vector_score=0.0, min_rerank_score=-50.0))
    t0 = datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc)
    fin_chunks = chunk_parsed_sections(
        [ParsedSection(text="Q3 revenue was 1,840,000 KWD.", page_number=1, source_filename="q3.pdf")],
        doc_id="doc-fin",
        file_type="pdf",
    )
    store.upsert_chunks(fin_chunks, total_pages=1, uploaded_at=t0)

    llm = FakeLLMClient(
        responses=[
            "Q3 revenue was 1,840,000 KWD [Source: q3.pdf | Page 1].",
            "This newly uploaded policy specifies 1 business day [Source: kyc.txt | Page 1].",
            "Q3 revenue was 1,840,000 KWD under new compliance rules [Source: q3.pdf | Page 1].",
        ]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    await rag.answer_question(QueryRequest(question="What is revenue?", session_id="mid-1"))

    # Upload a brand-new doc mid-conversation under same session_id
    kyc_chunks = chunk_parsed_sections(
        [ParsedSection(text="High-risk KYC escalation SLA is 1 business day.", page_number=1, source_filename="kyc.txt")],
        doc_id="doc-kyc",
        file_type="txt",
    )
    store.upsert_chunks(kyc_chunks, total_pages=1, uploaded_at=t0 + timedelta(minutes=2))

    resp2 = await rag.answer_question(
        QueryRequest(question="Summarize the newly uploaded doc please", session_id="mid-1")
    )
    assert resp2.is_found is True
    assert resp2.documents_used == ["kyc.txt"]
    assert "q3.pdf" not in str(llm.calls[1]["user_prompt"])
    assert "kyc.txt" in str(llm.calls[1]["user_prompt"])
    assert all("Conversation History" not in str(c["user_prompt"]) for c in llm.calls[:2])

    # A topic query containing "new" (not referring to a new file) must still search ALL docs.
    await rag.answer_question(
        QueryRequest(question="What does revenue under new compliance rules say?")
    )
    assert "q3.pdf" in str(llm.calls[2]["user_prompt"])


@pytest.mark.asyncio
async def test_empty_kb_short_circuits_without_any_llm_call_even_with_history() -> None:
    llm = FakeLLMClient()
    sessions = InMemorySessionStore()
    sessions.append_exchange("s-empty", "earlier question", "earlier answer")
    rag = RagService(vector_store=make_memory_vector_store(), llm_client=llm, session_store=sessions)

    resp = await rag.answer_question(QueryRequest(question="What is revenue?", session_id="s-empty"))

    assert resp.refusal_reason == "no_documents_indexed"
    # No follow-up condensing / generation is wasted when there is nothing to search.
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_stale_selected_doc_ids_are_reported_not_misreported_as_empty_kb() -> None:
    store = make_memory_vector_store(make_test_settings(min_vector_score=0.0, min_rerank_score=-50.0))
    _seed_store(store)
    llm = FakeLLMClient(responses=["Revenue was KWD 1,840,000 [1]."])
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    # Every selected doc was deleted server-side -> distinct reason, no search across all docs.
    gone = await rag.answer_question(
        QueryRequest(question="What was Q3 2026 revenue?", doc_ids=["deleted-doc"])
    )
    assert gone.refusal_reason == "selected_documents_missing"
    assert len(llm.calls) == 0

    # A mix of stale + valid IDs keeps working, scoped to the valid one only.
    mixed = await rag.answer_question(
        QueryRequest(question="What was Q3 2026 revenue?", doc_ids=["deleted-doc", "doc-fin"])
    )
    assert mixed.is_found is True
    assert all(c.doc_id == "doc-fin" for c in mixed.citations)


async def _collect_stream(rag: RagService, question: str) -> list[dict]:
    return [ev async for ev in rag.stream_question(QueryRequest(question=question))]


@pytest.mark.asyncio
async def test_stream_refusal_with_preamble_never_types_out_text() -> None:
    """Model writes prose then the sentinel: the UI must get no tokens and no sources."""
    store = make_memory_vector_store()
    _seed_store(store)
    llm = FakeLLMClient(
        stream_chunks=["The documents ", "do not mention ", "partnerships. ", "NO_RELEVANT_INFO_FOUND"]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    events = await _collect_stream(rag, "What was Q3 2026 consolidated revenue?")

    assert not [e for e in events if e["event"] == "token"]
    assert not [e for e in events if e["event"] == "citations"]
    done = events[-1]["data"]
    assert done["response_type"] == "not_found"
    assert done["final_answer"].startswith(NO_INFO_USER_MESSAGE)


@pytest.mark.asyncio
async def test_stream_does_not_retract_answer_when_sentinel_appears_late() -> None:
    """Once a cited answer is visible, a trailing sentinel is stripped, not turned into a refusal."""
    store = make_memory_vector_store()
    _seed_store(store)
    llm = FakeLLMClient(
        stream_chunks=["Revenue was KWD 1,840,000 [1]. ", "Partnerships are not covered. NO_RELEVANT_", "INFO_FOUND"]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())

    events = await _collect_stream(rag, "What was Q3 2026 consolidated revenue?")

    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert NO_INFO_SENTINEL not in streamed and "NO_RELEVANT_" not in streamed
    assert "1,840,000" in streamed
    types = [e["event"] for e in events]
    assert types.index("citations") < types.index("token")
    done = events[-1]["data"]
    assert done["is_found"] is True
    assert NO_INFO_SENTINEL not in done["final_answer"]


@pytest.mark.asyncio
async def test_stream_no_chunks_refusal_has_no_token_events() -> None:
    store = make_memory_vector_store()  # empty KB
    rag = RagService(
        vector_store=store, llm_client=FakeLLMClient(responses=[]), session_store=InMemorySessionStore()
    )
    events = await _collect_stream(rag, "What was Q3 revenue?")
    assert [e["event"] for e in events] == ["status", "done"]
    assert events[-1]["data"]["refusal_reason"] == "no_documents_indexed"
