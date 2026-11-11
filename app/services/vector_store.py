"""Qdrant vector store with local FastEmbed embeddings and cross-encoder reranking."""

import contextlib
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from app.config import Settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.schemas.document import DocumentInfo
from app.services.text_splitter import DocumentChunk

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_QUERY_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "about",
        "would",
        "could",
        "should",
        "shall",
        "into",
        "over",
        "such",
        "than",
        "then",
        "them",
        "they",
        "were",
        "been",
        "have",
        "has",
        "does",
        "done",
        "will",
        "what's",
        "tell",
        "explain",
        "describe",
        "give",
        "show",
        "list",
        "find",
        "mention",
        "state",
        "document",
        "documents",
        "pdf",
        "docx",
        "txt",
        "file",
        "files",
        "report",
    ]
)


def _extract_query_keywords(query: str) -> set[str]:
    """Return informative lowercase content tokens for hybrid lexical rescue."""
    return {tok for tok in _TOKEN_RE.findall(query.lower()) if tok not in _QUERY_STOPWORDS}


# Bounded lexical rescue: a chunk just under the rerank gate is still admitted when it shares
# a content keyword with the question and both scores are close to the gate. Pure noise
# scores around -11, so it can never qualify. Calibration is in the verification report.
_RESCUE_ENABLED_AT_OR_BELOW_GATE = -5.0  # only applies with a permissive configured gate
_RESCUE_MIN_COSINE = 0.48
_RESCUE_MIN_RERANK = -8.8
# Ranking bonus per shared keyword, capped so lexical overlap cannot outrank relevance.
_OVERLAP_BONUS_PER_KEYWORD = 0.8
_OVERLAP_BONUS_CAP = 2.5

_MAX_OVERVIEW_DOCS = 12
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WINDOW_SENTENCES = 2


def _sentence_windows(text: str, size: int = _WINDOW_SENTENCES) -> list[str]:
    """Overlapping windows of `size` consecutive sentences/lines (empty if the chunk is short).

    Small windows matter: a sentence grouped with neighbouring table rows is diluted again.
    """
    units = [u.strip() for u in _SENTENCE_SPLIT_RE.split(text) if len(u.strip()) > 3]
    if len(units) <= size:
        return []
    return [" ".join(units[i : i + size]) for i in range(len(units) - size + 1)]


def _lexical_overlap_count(keywords: set[str], text: str) -> int:
    """Count exact query keyword hits present inside a chunk."""
    if not keywords:
        return 0
    chunk_tokens = set(_TOKEN_RE.findall(text.lower()))
    return len(keywords & chunk_tokens)


@dataclass(slots=True, kw_only=True)
class RetrievedChunk:
    """Internal search result holding vector and cross-encoder scores."""

    chunk_id: str
    doc_id: str
    document_name: str
    page: int
    chunk_index: int
    text: str
    similarity_score: float
    rerank_score: float | None = None

    @classmethod
    def from_point(cls, point: Any, *, similarity_score: float) -> "RetrievedChunk":
        """Build a chunk from a Qdrant point and its stored payload."""
        payload = point.payload or {}
        return cls(
            chunk_id=str(point.id),
            doc_id=str(payload.get("doc_id", "")),
            document_name=str(payload.get("source_filename", "unknown")),
            page=int(payload.get("page_number", 1)),
            chunk_index=int(payload.get("chunk_index", 0)),
            text=str(payload.get("text", "")),
            similarity_score=similarity_score,
        )


class QdrantVectorStore:
    """Manages Qdrant storage, FastEmbed bi-encoder indexing, and cross-encoder reranking."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: QdrantClient | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self._settings = settings
        self._collection_name = settings.qdrant_collection
        self._embedding_model_name = settings.embedding_model
        self._is_e5_model = "e5" in settings.embedding_model.lower()

        self._client = client if client is not None else self._init_client(settings)
        self._embedder = embedder
        self._reranker = reranker
        self._vector_size: int | None = None

    @staticmethod
    def _init_client(settings: Settings) -> QdrantClient:
        try:
            if settings.qdrant_url.strip():
                logger.info("Connecting to remote Qdrant at %s", settings.qdrant_url)
                return QdrantClient(
                    url=settings.qdrant_url.strip(),
                    api_key=settings.qdrant_api_key.strip() or None,
                    timeout=30,
                )
            storage_path = Path(settings.qdrant_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            logger.info("Using local embedded Qdrant storage at %s", storage_path)
            return QdrantClient(path=str(storage_path))
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Qdrant client: {exc}") from exc

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
            logger.info("Loading FastEmbed embedding model: %s", self._embedding_model_name)
            self._embedder = TextEmbedding(
                model_name=self._embedding_model_name,
                cache_dir=cache_dir,
            )
        return self._embedder

    def _get_reranker(self) -> Any | None:
        if not self._settings.enable_reranker:
            return None
        if self._reranker is None:
            cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or None
            logger.info("Loading FastEmbed cross-encoder reranker: %s", self._settings.reranker_model)
            self._reranker = TextCrossEncoder(
                model_name=self._settings.reranker_model,
                cache_dir=cache_dir,
            )
        return self._reranker

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        prepared = [f"passage: {t}" for t in texts] if self._is_e5_model else texts
        return [list(map(float, vec)) for vec in embedder.embed(prepared)]

    def _embed_query(self, query: str) -> list[float]:
        embedder = self._get_embedder()
        prepared = f"query: {query}" if self._is_e5_model else query
        vec = next(iter(embedder.embed([prepared])))
        return [float(x) for x in vec]

    def _collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self._collection_name))
        except Exception as exc:
            raise VectorStoreError(f"Failed checking Qdrant collection: {exc}") from exc

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the cosine-distance collection on first upload, sized to the embedding model."""
        if self._collection_exists():
            return
        logger.info(
            "Creating Qdrant collection '%s' (dim=%d, COSINE)",
            self._collection_name,
            vector_size,
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        if self._settings.qdrant_url.strip():
            # Server Qdrant benefits from explicit payload indexes; local disk Qdrant ignores them.
            for field_name, schema in (
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("chunk_index", models.PayloadSchemaType.INTEGER),
                ("content_sha256", models.PayloadSchemaType.KEYWORD),
            ):
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=schema,
                    )
                except Exception as exc:
                    logger.warning("Could not create payload index '%s': %s", field_name, exc)

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        total_pages: int,
        uploaded_at: datetime | None = None,
        content_sha256: str | None = None,
    ) -> int:
        """Embed chunks and upsert them with full provenance metadata into Qdrant.

        `content_sha256` (hash of the original file bytes) is stored on every chunk so a
        re-upload of the same file can be detected without re-parsing or re-embedding it.
        """
        if not chunks:
            return 0

        timestamp = (uploaded_at or datetime.now(UTC)).isoformat()
        try:
            vectors = self._embed_passages([c.text for c in chunks])
            self._vector_size = len(vectors[0])
            self._ensure_collection(self._vector_size)

            points = [
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "doc_id": chunk.doc_id,
                        "source_filename": chunk.source_filename,
                        "file_type": chunk.file_type,
                        "page_number": chunk.page_number,
                        "total_pages": total_pages,
                        "chunk_index": chunk.chunk_index,
                        "uploaded_at": timestamp,
                        "content_sha256": content_sha256 or "",
                        "text": chunk.text,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            self._client.upsert(collection_name=self._collection_name, points=points, wait=True)
            return len(points)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to index chunks in Qdrant: {exc}") from exc

    def count_points(self, doc_ids: list[str] | None = None) -> int:
        """Return total indexed vectors (optionally scoped to `doc_ids`)."""
        if not self._collection_exists():
            return 0
        count_filter = self._build_doc_filter(doc_ids)
        try:
            result = self._client.count(
                collection_name=self._collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            raise VectorStoreError(f"Failed to count indexed document vectors: {exc}") from exc

    @staticmethod
    def _build_doc_filter(doc_ids: list[str] | None) -> models.Filter | None:
        if not doc_ids:
            return None
        return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])

    def search_and_rerank(
        self,
        query: str,
        *,
        top_k: int | None = None,
        rerank_k: int | None = None,
        min_vector_score: float | None = None,
        min_rerank_score: float | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run bi-encoder vector search followed by cosine gate and cross-encoder reranking."""
        if not self._collection_exists() or self.count_points(doc_ids) == 0:
            return []

        resolved_top_k = top_k if top_k is not None else self._settings.retrieval_top_k
        resolved_rerank_k = rerank_k if rerank_k is not None else self._settings.rerank_top_k
        vec_threshold = min_vector_score if min_vector_score is not None else self._settings.min_vector_score
        rerank_threshold = (
            min_rerank_score if min_rerank_score is not None else self._settings.min_rerank_score
        )

        try:
            query_vector = self._embed_query(query)
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                query_filter=self._build_doc_filter(doc_ids),
                limit=resolved_top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant similarity search failed: {exc}") from exc

        candidates = [
            RetrievedChunk.from_point(point, similarity_score=float(point.score))
            for point in response.points
            if float(point.score) >= vec_threshold
        ]

        if not candidates:
            logger.info("All retrieval candidates rejected by MIN_VECTOR_SCORE=%.3f", vec_threshold)
            return []

        reranker = self._get_reranker()
        if reranker is None:
            candidates.sort(key=lambda c: c.similarity_score, reverse=True)
            return candidates[:resolved_rerank_k]

        try:
            rerank_scores = [float(s) for s in reranker.rerank(query, [c.text for c in candidates])]
        except Exception as exc:
            raise VectorStoreError(f"Cross-encoder reranking failed: {exc}") from exc

        if self._settings.rerank_window_scoring:
            rerank_scores = self._max_window_scores(
                reranker, query, candidates, rerank_scores, below=rerank_threshold
            )

        query_keywords = _extract_query_keywords(query)
        allow_lexical_rescue = rerank_threshold <= _RESCUE_ENABLED_AT_OR_BELOW_GATE

        verified: list[tuple[float, RetrievedChunk]] = []
        for candidate, r_score in zip(candidates, rerank_scores, strict=True):
            candidate.rerank_score = r_score
            overlap = _lexical_overlap_count(query_keywords, candidate.text)
            rescued = (
                allow_lexical_rescue
                and overlap >= 1
                and candidate.similarity_score >= _RESCUE_MIN_COSINE
                and r_score >= _RESCUE_MIN_RERANK
            )
            if r_score >= rerank_threshold or rescued:
                bonus = min(_OVERLAP_BONUS_CAP, overlap * _OVERLAP_BONUS_PER_KEYWORD)
                verified.append((r_score + bonus, candidate))

        verified.sort(key=lambda item: (item[0], item[1].similarity_score), reverse=True)
        return [chunk for _, chunk in verified[:resolved_rerank_k]]

    @staticmethod
    def _max_window_scores(
        reranker: Any,
        query: str,
        candidates: list[RetrievedChunk],
        chunk_scores: list[float],
        *,
        below: float,
    ) -> list[float]:
        """Raise a rejected chunk's score to its best sentence window (one batched call).

        Only chunks scoring under the gate (`below`) are windowed: those are the ones a
        buried answer sentence can rescue, and chunks that already pass cost nothing extra.
        Window scoring can only raise a score, never lower it. Off-topic questions stay at
        noise level because no single sentence of the corpus matches them either.
        """
        owners: list[int] = []
        windows: list[str] = []
        for idx, candidate in enumerate(candidates):
            if chunk_scores[idx] >= below:
                continue
            for window in _sentence_windows(candidate.text):
                owners.append(idx)
                windows.append(window)
        if not windows:
            return chunk_scores
        try:
            window_scores = [float(s) for s in reranker.rerank(query, windows)]
        except Exception as exc:  # window scoring is an enhancement; keep chunk scores
            logger.warning("Sentence-window reranking failed, using chunk scores: %s", exc)
            return chunk_scores
        best = list(chunk_scores)
        for owner, score in zip(owners, window_scores, strict=True):
            best[owner] = max(best[owner], score)
        return best

    @property
    def default_rerank_k(self) -> int:
        """Return configured default maximum evidence chunks per query."""
        return self._settings.rerank_top_k

    def get_lead_chunks(
        self,
        *,
        limit: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return opening chunks per document (ordered by page/chunk) for broad overview queries."""
        if not self._collection_exists() or self.count_points(doc_ids) == 0:
            return []

        resolved_limit = limit if limit is not None else self._settings.rerank_top_k
        # chunk_index restarts at 0 for every document, so only each document's first
        # `resolved_limit` chunks can ever be selected; fetch exactly those, paginating
        # so large corpora never truncate the result set or miss documents.
        conditions: list[models.Condition] = [
            models.FieldCondition(key="chunk_index", range=models.Range(lt=resolved_limit))
        ]
        if doc_ids:
            conditions.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
        q_filter = models.Filter(must=conditions)
        records: list[Any] = []
        offset: Any = None
        try:
            while True:
                page, offset = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=q_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                records.extend(page)
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError(f"Failed to read document opening chunks: {exc}") from exc

        by_doc: dict[str, list[RetrievedChunk]] = {}
        uploaded_times: dict[str, str] = {}
        for point in records:
            chunk = RetrievedChunk.from_point(point, similarity_score=1.0)
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
            uploaded_at = str((point.payload or {}).get("uploaded_at", ""))
            uploaded_times.setdefault(chunk.doc_id, uploaded_at)

        for chunk_list in by_doc.values():
            chunk_list.sort(key=lambda c: (c.page, c.chunk_index))

        # Order newest uploads first so newly added docs mid-conversation take lead priority.
        ordered_doc_ids = sorted(
            by_doc.keys(),
            key=lambda did: uploaded_times.get(did, ""),
            reverse=True,
        )

        # Round-robin across selected documents so multi-doc summaries include every file.
        selected: list[RetrievedChunk] = []
        doc_queues = [by_doc[did] for did in ordered_doc_ids]
        # Every document gets at least its opening chunk, even when there are more documents
        # than the evidence budget ("summarize all documents" with 6 files and top_k=4).
        resolved_limit = max(resolved_limit, min(len(doc_queues), _MAX_OVERVIEW_DOCS))
        idx = 0
        while len(selected) < resolved_limit and doc_queues:
            active_queues: list[list[RetrievedChunk]] = []
            for queue in doc_queues:
                if idx < len(queue) and len(selected) < resolved_limit:
                    selected.append(queue[idx])
                if idx + 1 < len(queue):
                    active_queues.append(queue)
            doc_queues = active_queues
            idx += 1

        return selected

    def find_document_by_hash(self, content_sha256: str) -> DocumentInfo | None:
        """Return the indexed document whose original file bytes hash to `content_sha256`."""
        if not content_sha256 or not self._collection_exists():
            return None
        try:
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_sha256",
                            match=models.MatchValue(value=content_sha256),
                        )
                    ]
                ),
                limit=1,
                with_payload=["doc_id"],
                with_vectors=False,
            )
        except Exception as exc:
            # A failed duplicate check must never block an upload; index it normally.
            logger.warning("Duplicate-content lookup failed: %s", exc)
            return None
        if not records:
            return None
        doc_id = str((records[0].payload or {}).get("doc_id", ""))
        _, documents = self.list_documents()
        return next((d for d in documents if d.doc_id == doc_id), None)

    def list_documents(self) -> tuple[int, list[DocumentInfo]]:
        """Aggregate document-level metadata across all indexed chunks."""
        if not self._collection_exists():
            return 0, []

        grouped: dict[str, dict[str, Any]] = {}
        total_chunks = 0
        offset: Any = None

        try:
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=[
                        "doc_id",
                        "source_filename",
                        "file_type",
                        "page_number",
                        "total_pages",
                        "uploaded_at",
                    ],
                    with_vectors=False,
                )
                for record in records:
                    total_chunks += 1
                    payload = record.payload or {}
                    doc_id = str(payload.get("doc_id", ""))
                    if not doc_id:
                        continue
                    entry = grouped.get(doc_id)
                    if entry is None:
                        grouped[doc_id] = {
                            "doc_id": doc_id,
                            "filename": str(payload.get("source_filename", "unknown")),
                            "file_type": str(payload.get("file_type", "txt")),
                            "page_count": int(payload.get("total_pages", payload.get("page_number", 1))),
                            "chunk_count": 1,
                            "uploaded_at": str(payload.get("uploaded_at", datetime.now(UTC).isoformat())),
                        }
                    else:
                        entry["chunk_count"] += 1

                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            raise VectorStoreError(f"Failed to list indexed documents: {exc}") from exc

        documents = [
            DocumentInfo(
                doc_id=item["doc_id"],
                filename=item["filename"],
                file_type=item["file_type"],
                page_count=item["page_count"],
                chunk_count=item["chunk_count"],
                uploaded_at=datetime.fromisoformat(item["uploaded_at"]),
            )
            for item in grouped.values()
        ]
        documents.sort(key=lambda d: d.uploaded_at, reverse=True)
        return total_chunks, documents

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to `doc_id` and return removed vector count."""
        if not self._collection_exists():
            return 0
        doc_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        try:
            existing = int(
                self._client.count(
                    collection_name=self._collection_name,
                    count_filter=doc_filter,
                    exact=True,
                ).count
            )
            if existing == 0:
                return 0
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=doc_filter),
                wait=True,
            )
            return existing
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{doc_id}': {exc}") from exc

    def reset_collection(self) -> None:
        """Delete the entire vector collection so new sessions start clean."""
        try:
            if self._collection_exists():
                self._client.delete_collection(collection_name=self._collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset vector collection: {exc}") from exc

    def close(self) -> None:
        """Close Qdrant client cleanly to release local disk locks."""
        with contextlib.suppress(Exception):
            self._client.close()
