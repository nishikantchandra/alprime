"""Grounded question answering: scope, retrieval, generation and refusal handling.

The anti-hallucination layers, in order:
  1. An empty or stale document scope is refused before any search or model call.
  2-3. Retrieval drops passages below the cosine and cross-encoder relevance gates;
       if nothing survives, the question is refused without calling the model.
  4. Document text reaches the model inside XML fences (see `prompts.py`).
  5. The model can decline with a sentinel, which is turned into a templated refusal.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import QueryTrace, start_trace
from app.schemas.document import DocumentInfo
from app.schemas.query import Citation, QueryRequest, QueryResponse
from app.services.citations import (
    has_citation_marker,
    normalize_cited_chunks,
    sentinel_holdback,
    usable_answer,
)
from app.services.llm_client import LLMClient
from app.services.prompts import (
    GROUNDED_QA_SYSTEM_PROMPT,
    NO_INFO_SENTINEL,
    NO_INFO_USER_MESSAGE,
    QUERY_CONDENSE_SYSTEM_PROMPT,
    ContextChunkForPrompt,
    build_grounded_user_prompt,
)
from app.services.query_intent import (
    classify_conversational_intent,
    is_document_overview_query,
    needs_reasoning,
    resolve_recency_doc_ids,
    tokenize_words,
)
from app.services.replies import (
    CONVERSATIONAL_STATUS,
    REFUSAL_DETAILS,
    build_conversational_reply,
    build_refusal_message,
)
from app.services.session_store import ConversationTurn, InMemorySessionStore
from app.services.vector_store import QdrantVectorStore, RetrievedChunk

logger = get_logger(__name__)

# Streaming commit gate: tokens are held back until the answer is clearly a real answer
# (a citation marker appears, or enough text accumulates), so a refusal never "types out"
# and then flips into a "no info" message.
_STREAM_COMMIT_CHARS = 160
_CONDENSE_HISTORY_TURNS = 6
_CONDENSE_MAX_TOKENS = 320

Event = dict[str, Any]


@dataclass(slots=True)
class _DocScope:
    """Which documents a query may search, after stale selections are dropped."""

    doc_ids: tuple[str, ...]  # every document in scope, sorted (session history key)
    filenames: list[str]
    search_doc_ids: list[str] | None  # retrieval filter; None searches everything


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


def _relevance(chunk: RetrievedChunk) -> float:
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score


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
            if kept is None or _relevance(chunk) > _relevance(kept):
                best[chunk.chunk_id] = chunk
    merged = sorted(
        best.values(),
        key=lambda c: (c.rerank_score if c.rerank_score is not None else -999.0, c.similarity_score),
        reverse=True,
    )
    return merged[:limit]


def _trace_retrieval(trace: QueryTrace, prepared: _PreparedQuery) -> None:
    """Record retrieval outcome (scope, scores, rewrite) on the per-query trace line."""
    chunks = prepared.chunks
    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    trace.set(
        docs_in_scope=len(prepared.doc_scope),
        overview=prepared.is_overview,
        reasoning=prepared.reasoning,
        rewritten_query=prepared.retrieval_query != trace.question,
        chunks_retrieved=len(chunks),
        top_similarity=round(max(c.similarity_score for c in chunks), 4) if chunks else None,
        top_rerank=round(max(rerank_scores), 4) if rerank_scores else None,
    )


def _to_citations(chunks: list[RetrievedChunk]) -> tuple[list[str], list[Citation]]:
    """Unique document names (in rank order) and one API citation per chunk."""
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


def _citations_event(docs_used: list[str], citations: list[Citation]) -> Event:
    return {
        "event": "citations",
        "data": {"documents_used": docs_used, "citations": [c.model_dump() for c in citations]},
    }


def _status_event(stage: str, message: str) -> Event:
    return {"event": "status", "data": {"stage": stage, "message": message}}


def _token_event(text: str) -> Event:
    return {"event": "token", "data": {"text": text}}


class RagService:
    """Coordinates query rewriting, retrieval, reranking and strictly grounded synthesis."""

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

    # --- Public API -------------------------------------------------------------------

    async def answer_question(self, request: QueryRequest) -> QueryResponse:
        """Answer a question in one response (`POST /query`)."""
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

    async def stream_question(self, request: QueryRequest) -> AsyncIterator[Event]:
        """Stream SSE events (`status`, `citations`, `token`, `done`) for one question."""
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

    # --- Shared pipeline steps --------------------------------------------------------

    async def _resolve_scope(self, request: QueryRequest) -> _DocScope:
        """Work out which documents the query may search.

        Selected `doc_ids` that no longer exist on the server (deleted, or the knowledge base
        was reset from another tab or a script) are dropped, so a stale UI selection never
        silently widens or empties the search. If none of them exist, the scope is empty.
        """
        _, docs = await asyncio.to_thread(self._vector_store.list_documents)

        if request.doc_ids:
            known_ids = {d.doc_id for d in docs}
            valid_ids = [d for d in dict.fromkeys(request.doc_ids) if d in known_ids]
            stale = [d for d in request.doc_ids if d not in known_ids]
            if stale:
                logger.info("Ignoring %d stale selected doc_id(s): %s", len(stale), stale)
            search_ids: list[str] | None = valid_ids
            scope = tuple(sorted(valid_ids))
        else:
            search_ids = resolve_recency_doc_ids(request.question, docs)
            scope = tuple(sorted(d.doc_id for d in docs))

        scoped_docs: list[DocumentInfo] = (
            [d for d in docs if d.doc_id in search_ids] if search_ids is not None else docs
        )
        return _DocScope(scope, [d.filename for d in scoped_docs], search_ids)

    async def _condense_follow_up(self, question: str, history: list[ConversationTurn]) -> str:
        """Rewrite a multi-turn follow-up ("and for Q2?") into a standalone search query."""
        if not history or not self._llm_client.is_configured:
            return question

        # Earlier refusals add no useful context and can drag the rewrite off topic.
        useful_turns = [
            turn
            for turn in history[-_CONDENSE_HISTORY_TURNS:]
            if not (turn.role == "assistant" and NO_INFO_USER_MESSAGE in turn.content)
        ]
        if not useful_turns:
            return question

        formatted_history = "\n".join(f"{turn.role.capitalize()}: {turn.content}" for turn in useful_turns)
        condense_prompt = (
            f"Conversation History:\n{formatted_history}\n\nLatest Follow-up Question: {question}"
        )
        try:
            rewritten = await self._llm_client.generate(
                system_prompt=QUERY_CONDENSE_SYSTEM_PROMPT,
                user_prompt=condense_prompt,
                temperature=0.0,
                max_tokens=_CONDENSE_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("Query condensing failed, falling back to raw question: %s", exc)
            return question
        rewritten = rewritten.strip()
        return rewritten if len(rewritten) >= 4 else question

    async def _retrieve_evidence(
        self,
        *,
        is_overview: bool,
        retrieval_query: str,
        original_question: str,
        top_k: int | None,
        doc_ids: list[str] | None,
    ) -> list[RetrievedChunk]:
        """Semantic search with reranking; overview prompts also get each document's opening.

        When a follow-up was rewritten, both the rewrite and the user's original wording are
        searched and merged (best score per chunk). A rewrite that drifts can then no longer
        push the right passage out, and a vague original ("and the year before?") simply
        contributes nothing past the gates.
        """
        limit = top_k or self._vector_store.default_rerank_k
        queries = [retrieval_query]
        if original_question.strip() != retrieval_query.strip():
            queries.append(original_question)
        results = await asyncio.gather(
            *(
                asyncio.to_thread(self._vector_store.search_and_rerank, q, rerank_k=top_k, doc_ids=doc_ids)
                for q in queries
            )
        )
        semantic = _merge_ranked(list(results), limit=limit)
        if not is_overview:
            return semantic

        lead = await asyncio.to_thread(self._vector_store.get_lead_chunks, limit=limit, doc_ids=doc_ids)
        seen = {c.chunk_id for c in lead}
        return lead + [c for c in semantic if c.chunk_id not in seen]

    async def _prepare(self, request: QueryRequest, trace: QueryTrace) -> _PreparedQuery:
        """Resolve scope, rewrite follow-ups and retrieve evidence.

        `empty_reason` is set when there is nothing to search; in that case no LLM call or
        vector search is made at all.
        """
        with trace.span("scope"):
            scope = await self._resolve_scope(request)
            empty_reason = await self._empty_scope_reason(request, scope)
        if empty_reason:
            return _PreparedQuery(
                empty_reason, False, False, request.question, [], scope.doc_ids, scope.filenames
            )

        filename_words = set().union(*(tokenize_words(name) for name in scope.filenames))
        is_overview = is_document_overview_query(request.question, ignored_words=filename_words)
        history = (
            self._session_store.get_history(request.session_id, doc_scope=scope.doc_ids)
            if request.session_id and not is_overview
            else []
        )
        with trace.span("condense"):
            retrieval_query = await self._condense_follow_up(request.question, history)
        with trace.span("retrieve_rerank"):
            chunks = await self._retrieve_evidence(
                is_overview=is_overview,
                retrieval_query=retrieval_query,
                original_question=request.question,
                top_k=request.top_k,
                doc_ids=scope.search_doc_ids,
            )
        reasoning = self._reasoning_routing and not is_overview and needs_reasoning(retrieval_query)
        prepared = _PreparedQuery(
            None, is_overview, reasoning, retrieval_query, chunks, scope.doc_ids, scope.filenames
        )
        _trace_retrieval(trace, prepared)
        return prepared

    async def _empty_scope_reason(self, request: QueryRequest, scope: _DocScope) -> str | None:
        """Refusal reason when the scope holds no indexed chunks, otherwise None."""
        if scope.doc_ids and await self._count_points(scope.search_doc_ids or None) > 0:
            return None
        if request.doc_ids and await self._count_points(None) > 0:
            return "selected_documents_missing"
        return "no_documents_indexed"

    async def _count_points(self, doc_ids: list[str] | None) -> int:
        return await asyncio.to_thread(self._vector_store.count_points, doc_ids)

    async def _conversational_reply(self, request: QueryRequest) -> str | None:
        """Templated reply for pure small talk, recorded in the transcript; else None."""
        intent = classify_conversational_intent(request.question)
        if intent is None:
            return None
        scope = await self._resolve_scope(request)
        reply = build_conversational_reply(intent, scope.filenames)
        if request.session_id:
            self._session_store.append_transcript_only(
                request.session_id,
                request.question,
                reply,
                exchange_meta={
                    "response_type": "conversational",
                    "is_found": False,
                    "status_detail": CONVERSATIONAL_STATUS,
                },
            )
        return reply

    def _record(
        self,
        request: QueryRequest,
        answer: str,
        *,
        doc_scope: tuple[str, ...],
        refusal_reason: str | None = None,
        documents_used: list[str] | None = None,
        citations: list[Citation] | None = None,
    ) -> None:
        """Store the exchange in the session; `refusal_reason` marks it as not found."""
        if not request.session_id:
            return
        refused = refusal_reason is not None
        self._session_store.append_exchange(
            request.session_id,
            request.question,
            answer,
            doc_scope=doc_scope,
            exchange_meta={
                "response_type": "not_found" if refused else "grounded",
                "is_found": not refused,
                "refusal_reason": refusal_reason,
                "status_detail": REFUSAL_DETAILS[refusal_reason] if refused else None,
                "documents_used": documents_used or [],
                "citations": [c.model_dump() for c in (citations or [])],
            },
        )

    def _refuse(self, request: QueryRequest, prep: _PreparedQuery, reason: str) -> str:
        """Build and record a templated refusal; returns the message."""
        message = build_refusal_message(
            reason,
            searched_docs=prep.filenames,
            evidence_docs=[c.document_name for c in prep.chunks],
        )
        self._record(request, message, doc_scope=prep.doc_scope, refusal_reason=reason)
        return message

    @staticmethod
    def _grounded_prompt(request: QueryRequest, prep: _PreparedQuery) -> str:
        return build_grounded_user_prompt(
            request.question,
            [ContextChunkForPrompt(filename=c.document_name, page=c.page, text=c.text) for c in prep.chunks],
            standalone_question=prep.retrieval_query,
        )

    # --- One-shot answers -------------------------------------------------------------

    async def _answer_question(self, request: QueryRequest, trace: QueryTrace) -> QueryResponse:
        """Non-streaming pipeline: small talk -> retrieval gates -> LLM -> refusal check -> citations."""
        reply = await self._conversational_reply(request)
        if reply is not None:
            return QueryResponse(
                answer=reply,
                is_found=False,
                response_type="conversational",
                status_detail=CONVERSATIONAL_STATUS,
                session_id=request.session_id,
            )

        prep = await self._prepare(request, trace)
        if not prep.chunks:  # layers 1-3
            return self._refusal_response(request, prep, prep.empty_reason or "low_retrieval_relevance")

        with trace.span("llm"):
            raw_answer = await self._llm_client.generate(
                system_prompt=GROUNDED_QA_SYSTEM_PROMPT,
                user_prompt=self._grounded_prompt(request, prep),
                reasoning=prep.reasoning,
            )

        answer_text = usable_answer(raw_answer)
        if answer_text is None:  # layer 5
            return self._refusal_response(request, prep, "insufficient_document_evidence")

        clean_answer, cited_chunks = normalize_cited_chunks(answer_text, prep.chunks)
        docs_used, citations = _to_citations(cited_chunks)
        self._record(
            request,
            clean_answer,
            doc_scope=prep.doc_scope,
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

    def _refusal_response(self, request: QueryRequest, prep: _PreparedQuery, reason: str) -> QueryResponse:
        """Record the refusal in the session and wrap it as a `not_found` response."""
        return QueryResponse(
            answer=self._refuse(request, prep, reason),
            is_found=False,
            response_type="not_found",
            refusal_reason=reason,
            status_detail=REFUSAL_DETAILS[reason],
            session_id=request.session_id,
        )

    # --- Streaming answers ------------------------------------------------------------

    async def _stream_question(self, request: QueryRequest, trace: QueryTrace) -> AsyncIterator[Event]:
        """Streaming pipeline, emitted as SSE-ready events in this order:

        `status` (searching) -> `status` (generating) -> `citations` -> `token`* ->
        optional narrowed `citations` -> `done`. A refusal skips straight to `done`, so no
        partial answer text is ever shown for a question the documents cannot answer.
        """

        def done(
            final_answer: str,
            *,
            response_type: str,
            refusal_reason: str | None = None,
            status_detail: str | None = None,
        ) -> Event:
            return {
                "event": "done",
                "data": {
                    "is_found": response_type == "grounded",
                    "response_type": response_type,
                    "refusal_reason": refusal_reason,
                    "status_detail": status_detail,
                    "final_answer": final_answer,
                    "session_id": request.session_id,
                    "duration_ms": int((time.perf_counter() - trace.started) * 1000),
                },
            }

        def refusal(prep: _PreparedQuery, reason: str) -> Event:
            # Refusals arrive only in `done` (no token events), so the UI shows the badge and
            # message at once instead of typing out text it would then replace.
            return done(
                self._refuse(request, prep, reason),
                response_type="not_found",
                refusal_reason=reason,
                status_detail=REFUSAL_DETAILS[reason],
            )

        reply = await self._conversational_reply(request)
        if reply is not None:
            yield _citations_event([], [])
            yield _token_event(reply)
            yield done(reply, response_type="conversational", status_detail=CONVERSATIONAL_STATUS)
            return

        yield _status_event(
            "searching",
            "Reading opening sections of selected document(s)..."
            if is_document_overview_query(request.question)
            else "Searching & reranking uploaded documents...",
        )
        prep = await self._prepare(request, trace)
        if not prep.chunks:
            yield refusal(prep, prep.empty_reason or "low_retrieval_relevance")
            return

        docs_used, citations = _to_citations(prep.chunks)
        yield _status_event(
            "generating",
            f"Reasoning over {len(docs_used)} document(s)..."
            if prep.reasoning
            else f"Drafting cited answer from {len(docs_used)} document(s)...",
        )

        answer_text: str | None = None
        llm_started = time.perf_counter()
        async for event in self._stream_answer_tokens(request, prep, docs_used, citations):
            if event["event"] == "answer":
                answer_text = event["data"]
            else:
                yield event
        trace.timings_ms["llm"] = int((time.perf_counter() - llm_started) * 1000)

        if answer_text is None:
            yield refusal(prep, "insufficient_document_evidence")
            return

        # Narrow the citation list to the sources actually cited and renumber `[k]` markers.
        clean_text, cited_chunks = normalize_cited_chunks(answer_text, prep.chunks)
        cited_docs, cited = _to_citations(cited_chunks)
        if [c.chunk_id for c in cited_chunks] != [c.chunk_id for c in prep.chunks]:
            yield _citations_event(cited_docs, cited)

        self._record(
            request, clean_text, doc_scope=prep.doc_scope, documents_used=cited_docs, citations=cited
        )
        yield done(clean_text, response_type="grounded")

    async def _stream_answer_tokens(
        self,
        request: QueryRequest,
        prep: _PreparedQuery,
        docs_used: list[str],
        citations: list[Citation],
    ) -> AsyncIterator[Event]:
        """Relay model tokens behind the commit gate, then yield an internal `answer` event.

        Nothing is shown until the text is clearly an answer (a citation marker or enough
        characters), so a refusal is never typed out first. Once shown, the answer is never
        retracted; a stray sentinel is stripped. The final `answer` event carries the full
        text, or None when the model refused before anything was shown.
        """
        committed = False
        sent_len = 0  # characters of sentinel-free display text already sent
        collected: list[str] = []

        async for token in self._llm_client.stream(
            system_prompt=GROUNDED_QA_SYSTEM_PROMPT,
            user_prompt=self._grounded_prompt(request, prep),
            reasoning=prep.reasoning,
        ):
            collected.append(token)
            raw = "".join(collected)
            if not committed:
                head = raw.lstrip()
                if NO_INFO_SENTINEL in head:
                    break  # refusal decided before anything was shown: stop generating
                if not (has_citation_marker(head) or len(head) >= _STREAM_COMMIT_CHARS):
                    continue
                committed = True
                yield _citations_event(docs_used, citations)
                yield _status_event("writing", "Writing response...")
            display = raw.replace(NO_INFO_SENTINEL, "").lstrip()
            safe_end = len(display) - sentinel_holdback(display)
            if safe_end > sent_len:
                yield _token_event(display[sent_len:safe_end])
                sent_len = safe_end

        full_text = "".join(collected)
        if committed:
            display = full_text.replace(NO_INFO_SENTINEL, "").lstrip()
            if len(display) > sent_len:
                yield _token_event(display[sent_len:])
            yield {"event": "answer", "data": full_text.replace(NO_INFO_SENTINEL, "").strip()}
            return

        answer_text = usable_answer(full_text)
        if answer_text is not None:
            yield _citations_event(docs_used, citations)
            yield _status_event("writing", "Writing response...")
            yield _token_event(answer_text)
        yield {"event": "answer", "data": answer_text}
