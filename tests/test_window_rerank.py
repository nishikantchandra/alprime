"""Sentence-window reranking (MaxP) and overview-intent phrasing regressions from the trap audit."""

from app.config import Settings
from app.services.document_parser import ParsedSection
from app.services.query_intent import is_document_overview_query
from app.services.text_splitter import chunk_parsed_sections
from app.services.vector_store import _sentence_windows
from tests.conftest import make_memory_vector_store, make_test_settings

# A mixed-topic chunk: a fee table followed by the one sentence that answers the question.
_MIXED = (
    "Engagement Type | Fixed Fee Range | Typical Turnaround\n"
    "SME Market Feasibility Study | 2,500 - 4,200 KWD | 4 to 6 weeks\n"
    "Historical Ledger Cleanup | 1,200 KWD | 10 business days\n"
    "Invoices are payable net 15 calendar days via KNET corporate transfer or wire."
)


class _SentenceAwareReranker:
    """Scores a text high only if it contains 'payable'; long mixed texts are diluted."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        scores = []
        for doc in documents:
            if "payable" not in doc:
                scores.append(-11.0)
            else:  # dilution: the more unrelated lines around the answer, the lower the score
                scores.append(2.0 - 4.0 * doc.count("\n"))
        return scores


def _store(window_scoring: bool, reranker):
    settings = make_test_settings(
        min_vector_score=0.0, min_rerank_score=-6.0, rerank_window_scoring=window_scoring
    )
    store = make_memory_vector_store(settings, reranker=reranker)
    for doc_id, name, text in [
        ("doc-fee", "fees.docx", _MIXED),
        ("doc-pol", "policy.txt", "Standard risk onboarding review completes within 2 business days."),
    ]:
        chunks = chunk_parsed_sections(
            [ParsedSection(text=text, page_number=1, source_filename=name)],
            doc_id=doc_id,
            file_type="txt",
            chunk_size=800,
            chunk_overlap=50,
        )
        store.upsert_chunks(chunks, total_pages=1)
    return store


def test_window_scoring_is_on_by_default() -> None:
    assert Settings.model_fields["rerank_window_scoring"].default is True


def test_sentence_windows_skip_short_chunks() -> None:
    assert _sentence_windows("One sentence only.") == []
    assert _sentence_windows("A is one. B is two. C is three.") == [
        "A is one. B is two.",
        "B is two. C is three.",
    ]


def test_mixed_chunk_is_dropped_without_window_scoring() -> None:
    hits = _store(False, _SentenceAwareReranker()).search_and_rerank("What are the payment terms?")
    assert hits == []


def test_mixed_chunk_is_rescued_by_its_best_sentence_window() -> None:
    reranker = _SentenceAwareReranker()
    hits = _store(True, reranker).search_and_rerank("What are the payment terms?")
    assert [h.document_name for h in hits] == ["fees.docx"]
    assert hits[0].rerank_score is not None and hits[0].rerank_score > -6.0
    assert len(reranker.calls) == 2  # one chunk-level call + one batched window call


def test_window_scoring_never_admits_pure_noise() -> None:
    class _Noise:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [-11.0] * len(documents)

    assert _store(True, _Noise()).search_and_rerank("What is the capital of France?") == []


def test_window_reranker_failure_falls_back_to_chunk_scores() -> None:
    class _FailsOnWindows:
        def __init__(self) -> None:
            self.n = 0

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            self.n += 1
            if self.n > 1:
                raise RuntimeError("boom")
            return [-10.0] * len(documents)

    reranker = _FailsOnWindows()
    assert _store(True, reranker).search_and_rerank("payment terms") == []
    assert reranker.n == 2  # the window call was attempted, failed, and was swallowed


def test_chunks_that_already_pass_the_gate_are_not_windowed() -> None:
    class _AllPass:
        def __init__(self) -> None:
            self.calls = 0

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            self.calls += 1
            return [1.0] * len(documents)

    reranker = _AllPass()
    hits = _store(True, reranker).search_and_rerank("payment terms")
    assert len(hits) == 2 and reranker.calls == 1


def test_overview_lead_chunks_cover_every_document_beyond_top_k() -> None:
    settings = make_test_settings(min_vector_score=0.0, rerank_top_k=4)
    store = make_memory_vector_store(settings)
    for i in range(6):
        chunks = chunk_parsed_sections(
            [ParsedSection(text=f"Document {i} opening text.", page_number=1, source_filename=f"d{i}.txt")],
            doc_id=f"doc-{i}",
            file_type="txt",
            chunk_size=400,
            chunk_overlap=50,
        )
        store.upsert_chunks(chunks, total_pages=1)
    lead = store.get_lead_chunks(limit=4)
    assert {c.document_name for c in lead} == {f"d{i}.txt" for i in range(6)}


def test_default_evidence_budget_is_six_chunks() -> None:
    assert Settings.model_fields["rerank_top_k"].default == 6


def test_overview_detection_accepts_formatting_words() -> None:
    assert is_document_overview_query("Summarize all three documents in 3 bullets each.")
    assert is_document_overview_query("Give me a two sentence summary of each file")
    # A real topic still routes to semantic search.
    assert not is_document_overview_query("Summarize the overtime billing rules in 3 bullets")


# --------------------------------------------------------------------------- dual-query follow-ups
import pytest  # noqa: E402

from app.schemas.query import QueryRequest  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from app.services.session_store import InMemorySessionStore  # noqa: E402
from tests.conftest import FakeLLMClient  # noqa: E402


@pytest.mark.asyncio
async def test_drifted_follow_up_rewrite_still_retrieves_original_question_evidence() -> None:
    store = make_memory_vector_store(make_test_settings(min_vector_score=0.0))
    for doc_id, name, text in [
        ("doc-fin", "q3.pdf", "Days Sales Outstanding (DSO) improved to 34 days in Q3 2026."),
        ("doc-fee", "fees.docx", "Overtime beyond included hours is billed at KWD 35 per hour."),
    ]:
        chunks = chunk_parsed_sections(
            [ParsedSection(text=text, page_number=1, source_filename=name)],
            doc_id=doc_id,
            file_type="txt",
            chunk_size=400,
            chunk_overlap=50,
        )
        store.upsert_chunks(chunks, total_pages=1)
    llm = FakeLLMClient(
        responses=[
            "DSO was 34 days [1].",  # turn 1 answer
            "What was the DSO in Q3 2026?",  # turn 2 condense: drifts back to the old topic
            "Overtime is KWD 35 per hour [1].",  # turn 2 answer
        ]
    )
    rag = RagService(vector_store=store, llm_client=llm, session_store=InMemorySessionStore())
    await rag.answer_question(QueryRequest(question="What was the DSO?", session_id="s"))
    resp = await rag.answer_question(QueryRequest(question="What is the overtime rate?", session_id="s"))

    answer_prompt = str(llm.calls[-1]["user_prompt"])
    assert "KWD 35 per hour" in answer_prompt
    assert resp.is_found  # the fake answer cites [1]; the evidence check above is the point


def test_grounded_prompt_forbids_corpus_wide_absence_claims() -> None:
    from app.services.prompts import GROUNDED_QA_SYSTEM_PROMPT

    assert "provided passages do not state it" in GROUNDED_QA_SYSTEM_PROMPT
