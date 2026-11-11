"""Lightweight per-query tracing: one structured JSON log line per question.

This is a dependency-free stand-in for LangSmith/Langfuse-style tracing. Each query gets a
`QueryTrace` that records stage timings (condense, retrieval, LLM first token, total), the
retrieval scores that decided the outcome, which LLM provider actually served each call
(including failovers), and token usage when the provider reports it. When the query
finishes, the trace is emitted as a single `query_trace {...}` JSON line, ready to be
grepped, shipped to a log pipeline, or replaced later by an OpenTelemetry/Langfuse exporter.

The active trace lives in a `ContextVar`, so the LLM routing layer can attach provider and
token details without changing any client interface. Document text and answers are never
logged; the question is truncated.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger("app.trace")

_QUESTION_PREVIEW_CHARS = 120
_current_trace: ContextVar[QueryTrace | None] = ContextVar("current_query_trace", default=None)


def _ms(seconds: float) -> int:
    return int(round(seconds * 1000))


@dataclass
class QueryTrace:
    """Mutable record of one query's path through the RAG pipeline."""

    mode: str
    question: str
    session_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    timings_ms: dict[str, int] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    _emitted: bool = False

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a pipeline stage; repeated spans with the same name accumulate."""
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0) + _ms(time.perf_counter() - begin)

    def mark(self, name: str) -> None:
        """Record elapsed time since the query started (e.g. `first_token`), once."""
        self.timings_ms.setdefault(name, _ms(time.perf_counter() - self.started))

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def record_llm_call(self, **details: Any) -> None:
        self.llm_calls.append({k: v for k, v in details.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        question = " ".join(self.question.split())
        if len(question) > _QUESTION_PREVIEW_CHARS:
            question = question[: _QUESTION_PREVIEW_CHARS - 1] + "…"
        tokens = {
            key: sum(int(call.get(key) or 0) for call in self.llm_calls)
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
        }
        return {
            "trace_id": self.trace_id,
            "mode": self.mode,
            "session_id": self.session_id,
            "question": question,
            **self.attributes,
            "timings_ms": {**self.timings_ms, "total": _ms(time.perf_counter() - self.started)},
            "llm_calls": self.llm_calls,
            "tokens": tokens,
        }

    def emit(self) -> None:
        """Log the trace once; later calls are ignored."""
        if self._emitted:
            return
        self._emitted = True
        try:
            logger.info("query_trace %s", json.dumps(self.to_dict(), default=str, ensure_ascii=False))
        except Exception as exc:  # tracing must never break a request
            logger.warning("Failed to emit query trace: %s", exc)


def start_trace(*, mode: str, question: str, session_id: str | None) -> QueryTrace:
    trace = QueryTrace(mode=mode, question=question, session_id=session_id)
    _current_trace.set(trace)
    return trace


def current_trace() -> QueryTrace | None:
    return _current_trace.get()


def usage_fields(usage: Any) -> dict[str, int | None]:
    """Normalize OpenAI-style or Gemini-style usage objects into flat token counts."""
    if usage is None:
        return {}
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "prompt_token_count", None)
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "candidates_token_count", None)
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    if reasoning is None:
        reasoning = getattr(usage, "thoughts_token_count", None)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
    }


_pending_usage: ContextVar[dict[str, int | None] | None] = ContextVar("pending_llm_usage", default=None)


def report_usage(usage: Any) -> None:
    """Called by a concrete LLM client when the provider reports token usage."""
    fields = usage_fields(usage)
    if fields:
        _pending_usage.set(fields)


def take_usage() -> dict[str, int | None]:
    """Pop the usage reported by the call that just finished (empty when none was reported)."""
    fields = _pending_usage.get() or {}
    _pending_usage.set(None)
    return fields
