"""Core 5-layer anti-hallucination RAG orchestration service."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import html
import re
import time
from typing import Any, Literal

from app.core.logging import get_logger
from app.core.tracing import QueryTrace, start_trace
from app.schemas.document import DocumentInfo
from app.schemas.query import Citation, QueryRequest, QueryResponse
from app.services.llm_client import LLMClient
from app.services.prompts import (
    GROUNDED_QA_SYSTEM_PROMPT,
    NO_INFO_SENTINEL,
    NO_INFO_USER_MESSAGE,
    QUERY_CONDENSE_SYSTEM_PROMPT,
    ContextChunkForPrompt,
    build_grounded_user_prompt,
)
from app.services.session_store import ConversationTurn, InMemorySessionStore
from app.services.vector_store import QdrantVectorStore, RetrievedChunk

logger = get_logger(__name__)

# Streaming commit gate: tokens are held back until the answer is clearly a real answer
# (a citation marker appears, or enough text accumulates), so a refusal never "types out"
# and then flips into a "no info" message.
_STREAM_COMMIT_CHARS = 160
_CITATION_MARKER_RE = re.compile(r"\[\s*(?:\d+(?:\s*,\s*\d+)*|Source:)", re.IGNORECASE)


def _sentinel_holdback(text: str) -> int:
    """Length of the longest suffix of `text` that could be the start of the sentinel."""
    for size in range(min(len(text), len(NO_INFO_SENTINEL) - 1), 0, -1):
        if NO_INFO_SENTINEL.startswith(text[-size:]):
            return size
    return 0


def usable_answer(text: str) -> str | None:
    """Return the displayable answer, or None when the model refused.

    A bare sentinel (or empty output) is a refusal. If the model mixed the sentinel into a
    cited partial answer, the sentinel is stripped and the cited content is kept.
    """
    if NO_INFO_SENTINEL not in text:
        return text.strip() or None
    rest = text.replace(NO_INFO_SENTINEL, "").strip()
    return rest if _CITATION_MARKER_RE.search(rest) else None

# --- Overview (whole-document) intent detection -------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_SUMMARY_WORD_RE = re.compile(r"\b(summari[sz]e|summary|overview|synopsis|outline|gist)\b", re.IGNORECASE)
_DOC_NOUN_RE = re.compile(
    r"\b(doc|docs|document|documents|pdf|pdfs|file|files|report|reports|text|page|pages)\b",
    re.IGNORECASE,
)
_OVERVIEW_INTENT_RE = re.compile(
    r"\b(about|explain|describe|cover|covers|contain|contains|read|know|written|listed|stated|mentioned)\b",
    re.IGNORECASE,
)
# Words that carry no topic of their own in an overview request. If anything else is left
# over (e.g. "DSO", "KYC", "overtime"), the user is asking about a specific topic and the
# question is routed through normal semantic search instead.
_OVERVIEW_FILLER_WORDS = frozenset(
    """
    a an the this that these those it its my our your all each every both any some
    what whats what's is are was were be do does did can could would will should shall may might
    you me i we us please pls kindly now just quickly briefly short brief main key overall general
    give tell show provide write make get let help explain describe summarise summarize summary
    overview synopsis outline gist about of in on for to from with and or into within inside
    cover covers covered contain contains contained read know say says said talk talks discuss
    discusses mention mentions mentioned written list listed state stated everything detail
    details highlight highlights takeaway takeaways purpose scope asked asking mean means stuff
    things doc docs document documents pdf pdfs file files report reports text page pages
    uploaded selected attached provided given current whole entire content contents topic
    topics point points idea ideas high level high-level new newly latest newest recent most
    last added second first another next
    one two three four five six seven eight nine ten few couple several
    bullet bullets bulleted sentence sentences paragraph paragraphs line lines word words
    numbered format table form way style using use concise short-form
    """.split()
)

_LATEST_DOC_REF_RE = re.compile(
    r"\b(?:"
    r"(?:newly|just|latest|newest|last|most\s+recent(?:ly)?)\s+(?:uploaded|added)\s+"
    r"(?P<ext1>doc|document|file|pdf|docx|txt)"
    r"|"
    r"(?:latest|newest|most\s+recent)\s+(?P<ext2>doc|document|file|pdf|docx|txt)"
    r"|"
    r"(?:the|this|that)\s+new\s+(?:uploaded\s+|added\s+)?(?P<ext3>doc|document|file|pdf|docx|txt)"
    r")\b",
    re.IGNORECASE,
)


def _resolve_recency_doc_ids(question: str, docs: list[DocumentInfo]) -> list[str] | None:
    """Narrow retrieval to the newest upload when an unfiltered query targets a single latest doc."""
    if len(docs) <= 1:
        return None
    match = _LATEST_DOC_REF_RE.search(question)
    if not match:
        return None
    noun = (match.group("ext1") or match.group("ext2") or match.group("ext3") or "").lower()
    if noun in {"pdf", "docx", "txt"}:
        typed = [d for d in docs if d.file_type.lower() == noun]
        if typed:
            return [typed[0].doc_id]
    return [docs[0].doc_id]

_PUNCT_STRIP_RE = re.compile(r"[^\w\s']")
_GREETING_PHRASES = frozenset(
    {
        "hi", "hello", "hey", "hiya", "howdy", "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening", "how are you", "how are u",
        "how r u", "hows it going", "how's it going", "how is it going", "whats up",
        "what's up", "what is up", "hello how are you", "hi how are you",
        "hey how are you", "hi there how are you", "hello there how are you",
        "good morning how are you", "hi good morning", "hello good morning",
    }
)
_THANKS_PHRASES = frozenset(
    {
        "thanks", "thank you", "thank you very much", "thanks a lot", "thx", "ty",
        "ok", "okay", "got it", "great", "cool", "awesome", "nice", "bye", "goodbye",
    }
)
_HELP_PHRASES = frozenset(
    {"who are you", "what are you", "what can you do", "help", "how do you work"}
)

# Short "why" note shown under a refusal (badge tooltip / grey line in the UI).
_REFUSAL_DETAILS: dict[str, str] = {
    "no_documents_indexed": "The knowledge base is empty, so there was nothing to search.",
    "selected_documents_missing": (
        "The documents selected in the sidebar were deleted or the knowledge base was reset."
    ),
    "low_retrieval_relevance": (
        "No passage passed the relevance checks (vector similarity and reranker), "
        "so the question was not sent to the language model."
    ),
    "insufficient_document_evidence": (
        "Related passages were retrieved, but none of them states the answer, "
        "so the model declined rather than guess."
    ),
}
_MAX_NAMED_DOCS = 3


def _join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _describe_documents(filenames: list[str]) -> str:
    """'X.pdf', 'X.pdf and Y.docx', or 'your 5 uploaded documents'."""
    names = list(dict.fromkeys(filenames))
    if not names:
        return "your uploaded documents"
    if len(names) > _MAX_NAMED_DOCS:
        return f"your {len(names)} uploaded documents"
    return _join_names(names)


def build_refusal_message(
    reason: str,
    *,
    searched_docs: list[str] | None = None,
    evidence_docs: list[str] | None = None,
) -> str:
    """Natural "not found" reply: the fixed lead sentence plus what happened and what to do.

    Built from templates and real filenames only (never from the model), so a refusal can
    never introduce information that is not in the documents.
    """
    if reason == "no_documents_indexed":
        follow_up = (
            "There are no documents in the knowledge base right now, so there is nothing for me "
            "to answer from. Upload a PDF, DOCX, or TXT file and ask again."
        )
    elif reason == "selected_documents_missing":
        follow_up = (
            "The documents selected in the sidebar no longer exist on the server (they were "
            "deleted, or the knowledge base was reset). Refresh the document list, then "
            "re-select or re-upload them."
        )
    elif reason == "insufficient_document_evidence":
        where = _describe_documents(evidence_docs or searched_docs or [])
        follow_up = (
            f"The closest passages I found were in {where}, but they don't actually state this, "
            "so I won't guess or fill the gap with outside knowledge. If the detail is in another "
            "file, upload it and ask again, or rephrase the question using the document's own terms."
        )
    else:  # low_retrieval_relevance
        where = _describe_documents(searched_docs or [])
        follow_up = (
            f"I searched {where} and nothing in them is closely related to this question. "
            "I only answer from what the documents contain, so I won't respond from general "
            "knowledge. Ask about something they cover and I'll answer with page citations."
        )
    return f"{NO_INFO_USER_MESSAGE} {follow_up}"


@dataclass(slots=True)
class _PreparedQuery:
    """Everything the answer stage needs after scope resolution and retrieval."""

    empty_reason: str | None
    is_overview: bool
    reasoning: bool
    retrieval_query: str
    chunks: list[RetrievedChunk]
    doc_scope: tuple[str, ...]
    filenames: list[str] = field(default_factory=list)


def _trace_retrieval(trace: QueryTrace, prepared: _PreparedQuery) -> None:
    chunks = prepared.chunks
    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    trace.set(
        docs_in_scope=len(prepared.doc_scope),
        overview=prepared.is_overview,
        reasoning=prepared.reasoning,
        rewritten_query=prepared.retrieval_query != trace.question,
        chunks_retrieved=len(chunks),
        top_similarity=round(max((c.similarity_score for c in chunks), default=0.0), 4) if chunks else None,
        top_rerank=round(max(rerank_scores), 4) if rerank_scores else None,
    )


def classify_conversational_intent(question: str) -> Literal["greeting", "thanks", "help"] | None:
    """Detect pure conversational small-talk (greetings, thanks, role checks) safely."""
    normalized = " ".join(_PUNCT_STRIP_RE.sub(" ", question.lower()).split())
    if not normalized:
        return None
    if normalized in _GREETING_PHRASES:
        return "greeting"
    if normalized in _THANKS_PHRASES:
        return "thanks"
    if normalized in _HELP_PHRASES:
        return "help"
    return None

# --- Reasoning-needed detection ------------------------------------------------------
_REASONING_RE = re.compile(
    r"\b("
    r"compare|comparison|compared|versus|vs|difference|differ|differences|"
    r"calculate|calculation|compute|derive|estimate|"
    r"how\s+much\s+(?:more|less|higher|lower)|how\s+many\s+more|"
    r"percent(?:age)?\s+(?:change|increase|decrease|of)|growth|ratio|margin\s+change|"
    r"trend|trends|why|implication|implications|impact|consequence|"
    r"total|sum|average|combined|net\s+of|after\s+deduct\w*|"
    r"which\s+(?:is|was|one)\s+(?:higher|lower|larger|smaller|better|worse|cheaper|more|less)|"
    r"rank|step\s+by\s+step|reason|justify|evaluate|assess|"
    r"if\s+.+\s+(?:then|what|how)"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_WORD_RE = re.compile(r"\b(what|how|which|who|when|where|why)\b", re.IGNORECASE)
_SOURCE_TAG_RE = re.compile(r"\[Source:\s*(?P<name>[^|\]]+?)\s*\|\s*Page\s*(?P<page>\d+)\s*\]", re.IGNORECASE)
_INDEX_TAG_RE = re.compile(r"\[(?P<nums>\d+(?:\s*,\s*\d+)*)\](?!\()")


def is_document_overview_query(
    question: str,
    *,
    ignored_words: set[str] | None = None,
) -> bool:
    """Detect broad whole-document prompts ('summarize this report', 'what is this doc about').

    Returns False as soon as the question names a specific topic, so
    "Explain the DSO figure in the report" goes through semantic search.
    """
    cleaned = question.strip()
    if not cleaned:
        return False
    has_summary_word = bool(_SUMMARY_WORD_RE.search(cleaned))
    has_doc_intent = bool(_DOC_NOUN_RE.search(cleaned) and _OVERVIEW_INTENT_RE.search(cleaned))
    if not (has_summary_word or has_doc_intent):
        return False
    skip = _OVERVIEW_FILLER_WORDS | (ignored_words or set())
    topic_words = [
        w for w in _WORD_RE.findall(cleaned.lower()) if w not in skip and len(w) > 1
    ]
    return not topic_words


def needs_reasoning(question: str) -> bool:
    """Heuristic: comparisons, calculations, causal or multi-part questions need reasoning."""
    if _REASONING_RE.search(question):
        return True
    # Multi-part questions ("What is X and how does Y ...?", or several '?').
    return question.count("?") > 1 or len(_QUESTION_WORD_RE.findall(question)) >= 3


_CODE_WRAPPED_MARKER_RE = re.compile(r"`((?:\[\s*\d+(?:\s*,\s*\d+)*\s*\])+)`")


def unwrap_code_markers(answer: str) -> str:
    """Remove inline-code backticks some models put around citation markers (`[1]` -> [1])."""
    return _CODE_WRAPPED_MARKER_RE.sub(lambda m: m.group(1), answer)


def normalize_cited_chunks(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[RetrievedChunk]]:
    """Narrow chunks to cited sources and align compact `[k]` markers to output order.

    Supports both compact numeric `[1]` markers and legacy `[Source: <file> | Page <n>]` tags.
    """
    answer = unwrap_code_markers(answer)
    cited_legacy: set[tuple[str, int]] = set()
    for match in _SOURCE_TAG_RE.finditer(answer):
        name = html.unescape(match.group("name")).strip().lower()
        cited_legacy.add((name, int(match.group("page"))))
    if cited_legacy:
        kept = [c for c in chunks if (c.document_name.strip().lower(), c.page) in cited_legacy]
        return answer, (kept or chunks)

    valid_limit = len(chunks)
    if valid_limit == 0:
        return answer, chunks

    cited_order: list[int] = []
    seen_indices: set[int] = set()
    for match in _INDEX_TAG_RE.finditer(answer):
        nums = [int(part.strip()) for part in match.group("nums").split(",")]
        if not nums or any(n < 1 or n > valid_limit for n in nums):
            continue
        for n in nums:
            if n not in seen_indices:
                seen_indices.add(n)
                cited_order.append(n)

    if not cited_order:
        return answer, chunks

    kept = [chunks[idx - 1] for idx in cited_order]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(cited_order, start=1)}

    def _replace(match: re.Match[str]) -> str:
        nums = [int(part.strip()) for part in match.group("nums").split(",")]
        if not nums or any(n < 1 or n > valid_limit for n in nums):
            return match.group(0)
        deduped = list(dict.fromkeys(index_map[n] for n in nums))
        return "".join(f"[{mapped}]" for mapped in deduped)

    return _INDEX_TAG_RE.sub(_replace, answer), kept


def filter_cited_chunks(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep only the chunks the answer actually cited (or all chunks when untagged)."""
    _, kept = normalize_cited_chunks(answer, chunks)
    return kept


def _merge_ranked(result_lists: list[list[RetrievedChunk]], *, limit: int) -> list[RetrievedChunk]:
    """Union of ranked result lists, keeping each chunk's best score, best first.

    A single list is returned unchanged (its order already reflects the overlap bonus).
    """
    if len(result_lists) == 1:
        return result_lists[0][:limit]
    best: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for chunk in results:
            kept = best.get(chunk.chunk_id)
            score = chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score
            if kept is None or score > (
                kept.rerank_score if kept.rerank_score is not None else kept.similarity_score
            ):
                best[chunk.chunk_id] = chunk
    merged = sorted(
        best.values(),
        key=lambda c: (c.rerank_score if c.rerank_score is not None else -999.0, c.similarity_score),
        reverse=True,
    )
    return merged[:limit]


class RagService:
    """Coordinates query rewriting, vector retrieval, reranking, and strict synthesis."""

    def __init__(
        self,
        *,
        vector_store: QdrantVectorStore,
        llm_client: LLMClient,
        session_store: InMemorySessionStore,
        reasoning_routing: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._session_store = session_store
        self._reasoning_routing = reasoning_routing

    async def _condense_follow_up(
        self,
        question: str,
        history: list[ConversationTurn],
    ) -> str:
        """Rewrite multi-turn follow-up questions into standalone search queries."""
        if not history or not self._llm_client.is_configured:
            return question

        # Skip prior empty-refusal turns so unrelated misses do not skew query rewriting.
        useful_turns = [
            turn
            for turn in history[-6:]
            if not (turn.role == "assistant" and NO_INFO_USER_MESSAGE in turn.content)
        ]
        if not useful_turns:
            return question

        formatted_history = "\n".join(
            f"{turn.role.capitalize()}: {turn.content}" for turn in useful_turns
        )
        condense_prompt = (
            f"Conversation History:\n{formatted_history}\n\n"
            f"Latest Follow-up Question: {question}"
        )
        try:
            rewritten = (
                await self._llm_client.generate(
                    system_prompt=QUERY_CONDENSE_SYSTEM_PROMPT,
                    user_prompt=condense_prompt,
                    temperature=0.0,
                    max_tokens=320,
                )
            ).strip()
            if len(rewritten) < 4:
                return question
            return rewritten
        except Exception as exc:
            logger.warning("Query condensing failed, falling back to raw question: %s", exc)
            return question

    @staticmethod
    def _to_citations(chunks: list[RetrievedChunk]) -> tuple[list[str], list[Citation]]:
        docs_used = list(dict.fromkeys(c.document_name for c in chunks))
        citations = [
            Citation(
                document_name=c.document_name,
                doc_id=c.doc_id,
                page=c.page,
                similarity_score=round(c.similarity_score, 4),
                rerank_score=round(c.rerank_score, 4) if c.rerank_score is not None else None,
                snippet=c.text,
            )
            for c in chunks
        ]
        return docs_used, citations

    @staticmethod
    def _build_context_chunks(chunks: list[RetrievedChunk]) -> list[ContextChunkForPrompt]:
        return [
            ContextChunkForPrompt(filename=c.document_name, page=c.page, text=c.text)
            for c in chunks
        ]

    def _wants_reasoning(self, question: str, is_overview: bool) -> bool:
        return self._reasoning_routing and not is_overview and needs_reasoning(question)

    async def _retrieve_evidence(
        self,
        *,
        is_overview: bool,
        retrieval_query: str,
        top_k: int | None,
        doc_ids: list[str] | None,
        original_question: str | None = None,
    ) -> list[RetrievedChunk]:
        """Semantic+rerank search; overview prompts also get each document's opening chunks.

        When a follow-up was rewritten, both the rewrite and the user's original wording are
        searched and merged (best score per chunk). A rewrite that drifts (adds context to an
        already standalone question) can then no longer push the right passage out, and a
        vague original ("and the year before?") simply contributes nothing past the gates.
        """
        queries = [retrieval_query]
        if original_question and original_question.strip() != retrieval_query.strip():
            queries.append(original_question)
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._vector_store.search_and_rerank, q, rerank_k=top_k, doc_ids=doc_ids
                )
                for q in queries
            )
        )
        semantic = _merge_ranked(results, limit=top_k or self._vector_store.default_rerank_k)
        if not is_overview:
            return semantic

        lead = await asyncio.to_thread(
            self._vector_store.get_lead_chunks,
            limit=top_k or self._vector_store.default_rerank_k,
            doc_ids=doc_ids,
        )
        seen = {c.chunk_id for c in lead}
        return lead + [c for c in semantic if c.chunk_id not in seen]

    async def _doc_context_meta(
        self, request: QueryRequest
    ) -> tuple[tuple[str, ...], list[str], set[str], list[str] | None]:
        """Return active doc-ID scope, filenames, filename tokens, and effective retrieval IDs.

        Selected `doc_ids` that no longer exist on the server (deleted / KB reset from another
        tab, Swagger, or a script) are dropped, so a stale UI selection never silently empties
        the search scope. If none of them exist, the scope is empty.
        """
        _, docs = await asyncio.to_thread(self._vector_store.list_documents)

        if request.doc_ids:
            known_ids = {d.doc_id for d in docs}
            valid_ids = [d for d in dict.fromkeys(request.doc_ids) if d in known_ids]
            stale = [d for d in request.doc_ids if d not in known_ids]
            if stale:
                logger.info("Ignoring %d stale selected doc_id(s): %s", len(stale), stale)
            scope = tuple(sorted(valid_ids))
            effective_doc_ids: list[str] | None = valid_ids
            scoped_docs = [d for d in docs if d.doc_id in valid_ids]
        else:
            scope = tuple(sorted(d.doc_id for d in docs)) if docs else ()
            effective_doc_ids = _resolve_recency_doc_ids(request.question, docs)
            scoped_docs = (
                [d for d in docs if d.doc_id in effective_doc_ids]
                if effective_doc_ids
                else docs
            )

        filenames = [d.filename for d in scoped_docs]
        filename_words: set[str] = set()
        for name in filenames:
            filename_words.update(_WORD_RE.findall(name.lower()))
        return scope, filenames, filename_words, effective_doc_ids

    @staticmethod
    def _build_conversational_reply(
        intent: Literal["greeting", "thanks", "help"],
        filenames: list[str],
    ) -> str:
        if intent == "thanks":
            return (
                "You're welcome! Let me know if you have any other questions about your uploaded documents."
            )
        if not filenames:
            return (
                "Hello! I'm the AlPrime Document RAG Assistant. "
                "Please upload PDF, DOCX, or TXT files using the paperclip button below or the Document Vault, "
                "and ask me anything grounded strictly in their contents."
            )
        preview = ", ".join(filenames[:3])
        extra = f" (+{len(filenames) - 3} more)" if len(filenames) > 3 else ""
        if intent == "help":
            return (
                f"I answer questions strictly from your uploaded documents with page citations. "
                f"Currently loaded ({len(filenames)}): {preview}{extra}. "
                f"Ask a specific question or ask me to summarize them."
            )
        return (
            f"Hello! I'm doing well and ready to help. "
            f"You currently have {len(filenames)} document(s) loaded ({preview}{extra}). "
            f"Ask me any specific question or ask for a summary—all answers stay grounded in your uploaded files."
        )

    async def _prepare(self, request: QueryRequest, trace: QueryTrace) -> _PreparedQuery:
        """Resolve scope, rewrite follow-ups, and retrieve evidence.

        `empty_reason` is set when there is nothing to search; in that case no LLM call or
        vector search is made at all.
        """
        with trace.span("scope"):
            doc_scope, filenames, filename_words, effective_doc_ids = await self._doc_context_meta(
                request
            )
            has_docs = bool(doc_scope) and await asyncio.to_thread(
                self._vector_store.count_points, effective_doc_ids or None
            ) > 0
            if not has_docs:
                selection_went_stale = bool(request.doc_ids) and await asyncio.to_thread(
                    self._vector_store.count_points, None
                ) > 0
        if not has_docs:
            reason = "selected_documents_missing" if selection_went_stale else "no_documents_indexed"
            return _PreparedQuery(reason, False, False, request.question, [], doc_scope, filenames)

        is_overview = is_document_overview_query(request.question, ignored_words=filename_words)
        history = (
            self._session_store.get_history(request.session_id, doc_scope=doc_scope)
            if (request.session_id and not is_overview)
            else []
        )
        with trace.span("condense"):
            retrieval_query = await self._condense_follow_up(request.question, history)
        with trace.span("retrieve_rerank"):
            chunks = await self._retrieve_evidence(
                is_overview=is_overview,
                retrieval_query=retrieval_query,
                top_k=request.top_k,
                doc_ids=effective_doc_ids,
                original_question=request.question,
            )
        reasoning = self._wants_reasoning(retrieval_query, is_overview)
        prepared = _PreparedQuery(
            None, is_overview, reasoning, retrieval_query, chunks, doc_scope, filenames
        )
        _trace_retrieval(trace, prepared)
        return prepared

    def _record(
        self,
        request: QueryRequest,
        answer: str,
        *,
        doc_scope: tuple[str, ...] | None = None,
        response_type: str = "grounded",
        is_found: bool = True,
        refusal_reason: str | None = None,
        status_detail: str | None = None,
        documents_used: list[str] | None = None,
        citations: list[Citation] | None = None,
    ) -> None:
        if request.session_id:
            self._session_store.append_exchange(
                request.session_id,
                request.question,
                answer,
                doc_scope=doc_scope,
                exchange_meta={
                    "response_type": response_type,
                    "is_found": is_found,
                    "refusal_reason": refusal_reason,
                    "status_detail": status_detail,
                    "documents_used": documents_used or [],
                    "citations": [c.model_dump() for c in (citations or [])],
                },
            )

    def _record_conversational(self, request: QueryRequest, reply: str) -> None:
        if request.session_id:
            self._session_store.append_transcript_only(
                request.session_id,
                request.question,
                reply,
                exchange_meta={
                    "response_type": "conversational",
                    "is_found": False,
                    "status_detail": "Conversational greeting handled without document retrieval.",
                },
            )

    def _not_found(
        self,
        request: QueryRequest,
        *,
        reason: str,
        doc_scope: tuple[str, ...] | None = None,
        message: str | None = None,
    ) -> QueryResponse:
        detail = _REFUSAL_DETAILS[reason]
        message = message or build_refusal_message(reason)
        self._record(
            request,
            message,
            doc_scope=doc_scope,
            response_type="not_found",
            is_found=False,
            refusal_reason=reason,
            status_detail=detail,
        )
        return QueryResponse(
            answer=message,
            is_found=False,
            response_type="not_found",
            refusal_reason=reason,
            status_detail=detail,
            documents_used=[],
            citations=[],
            session_id=request.session_id,
        )

    async def answer_question(self, request: QueryRequest) -> QueryResponse:
        """Execute conversational guard or the 5-layer grounded RAG pipeline."""
        trace = start_trace(mode="sync", question=request.question, session_id=request.session_id)
        try:
            response = await self._answer_question(request, trace)
        except Exception as exc:
            trace.set(outcome="error", error=type(exc).__name__)
            raise
        else:
            trace.set(
                outcome=response.response_type,
                refusal_reason=response.refusal_reason,
                chunks_cited=len(response.citations),
            )
            return response
        finally:
            trace.emit()

    async def _answer_question(self, request: QueryRequest, trace: QueryTrace) -> QueryResponse:
        conv_intent = classify_conversational_intent(request.question)
        if conv_intent is not None:
            _scope, filenames, _words, _eff = await self._doc_context_meta(request)
            reply = self._build_conversational_reply(conv_intent, filenames)
            self._record_conversational(request, reply)
            return QueryResponse(
                answer=reply,
                is_found=False,
                response_type="conversational",
                status_detail="Conversational greeting handled without document retrieval.",
                documents_used=[],
                citations=[],
                session_id=request.session_id,
            )

        prep = await self._prepare(request, trace)
        chunks = prep.chunks

        # Layers 1-3 rejection: nothing indexed/selected, or no chunks cleared score thresholds.
        if not chunks:
            reason = prep.empty_reason or "low_retrieval_relevance"
            return self._not_found(
                request,
                reason=reason,
                doc_scope=prep.doc_scope,
                message=build_refusal_message(reason, searched_docs=prep.filenames),
            )

        user_prompt = build_grounded_user_prompt(
            request.question,
            self._build_context_chunks(chunks),
            standalone_question=prep.retrieval_query,
        )
        with trace.span("llm"):
            raw_answer = await self._llm_client.generate(
                system_prompt=GROUNDED_QA_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                reasoning=prep.reasoning,
            )

        # Layer 5 verification: honor model's negative sentinel if evidence was insufficient.
        answer_text = usable_answer(raw_answer)
        if answer_text is None:
            return self._not_found(
                request,
                reason="insufficient_document_evidence",
                doc_scope=prep.doc_scope,
                message=build_refusal_message(
                    "insufficient_document_evidence",
                    searched_docs=prep.filenames,
                    evidence_docs=[c.document_name for c in chunks],
                ),
            )

        clean_answer, cited_chunks = normalize_cited_chunks(answer_text, chunks)
        docs_used, citations = self._to_citations(cited_chunks)
        self._record(
            request,
            clean_answer,
            doc_scope=prep.doc_scope,
            response_type="grounded",
            is_found=True,
            documents_used=docs_used,
            citations=citations,
        )
        return QueryResponse(
            answer=clean_answer,
            is_found=True,
            response_type="grounded",
            documents_used=docs_used,
            citations=citations,
            session_id=request.session_id,
        )

    async def stream_question(self, request: QueryRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream SSE events (`status`, `citations`, `token`, `done`) with sentinel safety buffering."""
        trace = start_trace(mode="stream", question=request.question, session_id=request.session_id)
        try:
            async for event in self._stream_question(request, trace):
                if event["event"] == "token":
                    trace.mark("first_token")
                elif event["event"] == "done":
                    data = event["data"]
                    trace.set(
                        outcome=data.get("response_type"),
                        refusal_reason=data.get("refusal_reason"),
                    )
                elif event["event"] == "citations":
                    trace.set(chunks_cited=len(event["data"].get("citations", [])))
                yield event
        except Exception as exc:
            trace.set(outcome="error", error=type(exc).__name__)
            raise
        except BaseException:
            # Client closed the connection (browser stop / tab closed) before `done`.
            trace.set(outcome="client_disconnected")
            raise
        finally:
            trace.emit()

    async def _stream_question(
        self, request: QueryRequest, trace: QueryTrace
    ) -> AsyncIterator[dict[str, Any]]:
        started = trace.started

        def _done(
            is_found: bool,
            final_answer: str,
            *,
            response_type: str = "grounded",
            refusal_reason: str | None = None,
            status_detail: str | None = None,
        ) -> dict[str, Any]:
            return {
                "event": "done",
                "data": {
                    "is_found": is_found,
                    "response_type": response_type,
                    "refusal_reason": refusal_reason,
                    "status_detail": status_detail,
                    "final_answer": final_answer,
                    "session_id": request.session_id,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            }

        conv_intent = classify_conversational_intent(request.question)
        if conv_intent is not None:
            _scope, filenames, _words, _eff = await self._doc_context_meta(request)
            reply = self._build_conversational_reply(conv_intent, filenames)
            self._record_conversational(request, reply)
            yield {"event": "citations", "data": {"documents_used": [], "citations": []}}
            yield {"event": "token", "data": {"text": reply}}
            yield _done(
                False,
                reply,
                response_type="conversational",
                status_detail="Conversational greeting handled without document retrieval.",
            )
            return

        is_overview_hint = is_document_overview_query(request.question)
        yield {
            "event": "status",
            "data": {
                "stage": "searching",
                "message": (
                    "Reading opening sections of selected document(s)..."
                    if is_overview_hint
                    else "Searching & reranking uploaded documents..."
                ),
            },
        }

        prep = await self._prepare(request, trace)
        reasoning, retrieval_query, chunks, doc_scope = (
            prep.reasoning,
            prep.retrieval_query,
            prep.chunks,
            prep.doc_scope,
        )

        def _refusal(reason: str) -> dict[str, Any]:
            # Refusals are delivered only in `done` (no token events), so the UI shows the
            # badge + message at once instead of typing out text it will then replace.
            detail = _REFUSAL_DETAILS[reason]
            message = build_refusal_message(
                reason,
                searched_docs=prep.filenames,
                evidence_docs=[c.document_name for c in chunks],
            )
            self._record(
                request,
                message,
                doc_scope=doc_scope,
                response_type="not_found",
                is_found=False,
                refusal_reason=reason,
                status_detail=detail,
            )
            return _done(
                False,
                message,
                response_type="not_found",
                refusal_reason=reason,
                status_detail=detail,
            )

        if not chunks:
            yield _refusal(prep.empty_reason or "low_retrieval_relevance")
            return

        docs_used, citations = self._to_citations(chunks)
        yield {
            "event": "status",
            "data": {
                "stage": "generating",
                "message": (
                    f"Reasoning over {len(docs_used)} document(s)..."
                    if reasoning
                    else f"Drafting cited answer from {len(docs_used)} document(s)..."
                ),
            },
        }

        user_prompt = build_grounded_user_prompt(
            request.question,
            self._build_context_chunks(chunks),
            standalone_question=retrieval_query,
        )

        collected: list[str] = []
        committed = False  # once True, the answer is being shown and will not be retracted
        sent_len = 0  # chars of the sentinel-free display text already sent

        def _commit_events() -> list[dict[str, Any]]:
            return [
                {
                    "event": "citations",
                    "data": {
                        "documents_used": docs_used,
                        "citations": [c.model_dump() for c in citations],
                    },
                },
                {"event": "status", "data": {"stage": "writing", "message": "Writing response..."}},
            ]

        llm_started = time.perf_counter()
        async for token in self._llm_client.stream(
            system_prompt=GROUNDED_QA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            reasoning=reasoning,
        ):
            collected.append(token)
            raw = "".join(collected)
            if not committed:
                head = raw.lstrip()
                if NO_INFO_SENTINEL in head:
                    # Refusal decided before anything was shown: stop generation early.
                    break
                if not (_CITATION_MARKER_RE.search(head) or len(head) >= _STREAM_COMMIT_CHARS):
                    continue
                committed = True
                for ev in _commit_events():
                    yield ev
            # Strip any sentinel and hold back a possible partial sentinel at the tail.
            display = raw.replace(NO_INFO_SENTINEL, "").lstrip()
            safe_end = len(display) - _sentinel_holdback(display)
            if safe_end > sent_len:
                yield {"event": "token", "data": {"text": display[sent_len:safe_end]}}
                sent_len = safe_end

        trace.timings_ms["llm"] = int((time.perf_counter() - llm_started) * 1000)
        full_text = "".join(collected)
        if not committed:
            answer_text = usable_answer(full_text)
            if answer_text is None:
                yield _refusal("insufficient_document_evidence")
                return
            for ev in _commit_events():
                yield ev
            yield {"event": "token", "data": {"text": answer_text}}
        else:
            # Already visible to the user: never flip to a refusal; keep the partial answer.
            answer_text = full_text.replace(NO_INFO_SENTINEL, "").strip()
            display = full_text.replace(NO_INFO_SENTINEL, "").lstrip()
            if len(display) > sent_len:
                yield {"event": "token", "data": {"text": display[sent_len:]}}

        # Narrow the citation list to the sources the answer actually cited and align `[k]` markers.
        clean_text, cited_chunks = normalize_cited_chunks(answer_text, chunks)
        cited_docs, cited = self._to_citations(cited_chunks)
        if [c.chunk_id for c in cited_chunks] != [c.chunk_id for c in chunks]:
            yield {
                "event": "citations",
                "data": {"documents_used": cited_docs, "citations": [c.model_dump() for c in cited]},
            }

        self._record(
            request,
            clean_text,
            doc_scope=doc_scope,
            response_type="grounded",
            is_found=True,
            documents_used=cited_docs,
            citations=cited,
        )
        yield _done(True, clean_text, response_type="grounded")
