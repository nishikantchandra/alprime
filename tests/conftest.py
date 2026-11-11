"""Shared deterministic offline test doubles for Qdrant vector store and LLMs."""

from collections.abc import AsyncIterator
import math
import re

from qdrant_client import QdrantClient

from app.config import Settings
from app.core.exceptions import LLMConfigurationError
from app.services.llm_client import LLMClient
from app.services.vector_store import QdrantVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicEmbedder:
    """Lightweight offline bag-of-tokens embedder producing unit vectors of dim 64."""

    DIM = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            tokens = _TOKEN_RE.findall(text.lower())
            for tok in tokens:
                idx = sum(ord(ch) * (i + 1) for i, ch in enumerate(tok)) % self.DIM
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FakeCrossEncoder:
    """Offline lexical-overlap cross-encoder stub returning controllable scores."""

    def __init__(self, forced_scores: list[float] | None = None) -> None:
        self.forced_scores = forced_scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self.forced_scores is not None:
            return self.forced_scores[: len(documents)]
        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        scores: list[float] = []
        for doc in documents:
            d_tokens = set(_TOKEN_RE.findall(doc.lower()))
            overlap = len(q_tokens & d_tokens)
            scores.append(float(overlap * 2 - 1))
        return scores


class FakeLLMClient(LLMClient):
    """Configurable offline LLM stub recording calls for prompt/guardrail assertions."""

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        configured: bool = True,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self._responses = list(responses) if responses is not None else ["Default grounded answer."]
        self._configured = configured
        self._stream_chunks = stream_chunks
        self.calls: list[dict[str, object]] = []

    @property
    def provider_name(self) -> str:
        return "fake-llm (test)"

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> str:
        if not self._configured:
            raise LLMConfigurationError("Fake LLM unconfigured.")
        self.calls.append(
            {
                "mode": "generate",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning": reasoning,
            }
        )
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    async def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        if not self._configured:
            raise LLMConfigurationError("Fake LLM unconfigured.")
        self.calls.append(
            {
                "mode": "stream",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "reasoning": reasoning,
            }
        )
        chunks = (
            self._stream_chunks
            if self._stream_chunks is not None
            else [self._responses[0]]
        )
        for piece in chunks:
            yield piece


def make_test_settings(**overrides: object) -> Settings:
    """Create isolated Settings safe for fast in-memory testing."""
    defaults: dict[str, object] = {
        "qdrant_url": "",
        "qdrant_path": "./data/test_unused",
        "qdrant_collection": "test_collection",
        "min_vector_score": 0.25,
        "min_rerank_score": 0.0,
        "enable_reranker": True,
        # Position-forced fake scores map to chunks; window scoring is tested separately.
        "rerank_window_scoring": False,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "auth_enabled": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_memory_vector_store(
    settings: Settings | None = None,
    *,
    reranker: FakeCrossEncoder | None = None,
) -> QdrantVectorStore:
    """Build QdrantVectorStore backed by :memory: Qdrant + offline stubs."""
    cfg = settings or make_test_settings()
    return QdrantVectorStore(
        cfg,
        client=QdrantClient(":memory:"),
        embedder=DeterministicEmbedder(),
        reranker=reranker if reranker is not None else FakeCrossEncoder(),
    )
