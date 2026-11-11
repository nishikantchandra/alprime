"""In-memory multi-turn conversation history and transcript store keyed by session_id."""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Single conversation turn stored for follow-up query rewriting."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class TranscriptExchange:
    """Rich recorded Q&A turn stored for full conversation transcript export."""

    turn: int
    timestamp: datetime
    question: str
    answer: str
    response_type: str = "grounded"
    is_found: bool = True
    refusal_reason: str | None = None
    status_detail: str | None = None
    documents_used: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


class InMemorySessionStore:
    """Thread-safe LRU session store separating prompt context from full export transcripts."""

    def __init__(
        self,
        max_sessions: int = 500,
        max_turns_per_session: int = 10,
        max_transcript_turns: int = 100,
    ) -> None:
        self._max_sessions = max_sessions
        self._max_turns = max_turns_per_session
        self._max_transcript_turns = max_transcript_turns
        self._sessions: OrderedDict[str, tuple[tuple[str, ...] | None, list[ConversationTurn]]] = (
            OrderedDict()
        )
        self._transcripts: OrderedDict[str, list[TranscriptExchange]] = OrderedDict()
        self._lock = Lock()

    def _append_transcript_locked(
        self,
        session_id: str,
        user_question: str,
        assistant_answer: str,
        exchange_meta: dict[str, Any] | None,
    ) -> None:
        if session_id not in self._transcripts and len(self._transcripts) >= self._max_sessions:
            self._transcripts.popitem(last=False)
        items = self._transcripts.get(session_id, [])
        meta = exchange_meta or {}
        next_turn = (items[-1].turn + 1) if items else 1
        items.append(
            TranscriptExchange(
                turn=next_turn,
                timestamp=datetime.now(timezone.utc),
                question=user_question,
                answer=assistant_answer,
                response_type=str(meta.get("response_type", "grounded")),
                is_found=bool(meta.get("is_found", True)),
                refusal_reason=meta.get("refusal_reason"),
                status_detail=meta.get("status_detail"),
                documents_used=list(meta.get("documents_used") or []),
                citations=[dict(c) for c in (meta.get("citations") or [])],
            )
        )
        if len(items) > self._max_transcript_turns:
            del items[: len(items) - self._max_transcript_turns]
        self._transcripts[session_id] = items
        self._transcripts.move_to_end(session_id)

    def get_history(
        self,
        session_id: str,
        *,
        doc_scope: tuple[str, ...] | None = None,
    ) -> list[ConversationTurn]:
        """Return turns for `session_id`, invalidating stale turns if document scope changed."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return []
            stored_scope, turns = entry
            if doc_scope is not None and stored_scope is not None and stored_scope != doc_scope:
                # Documents were replaced/reset or user switched selected docs; start clean.
                self._sessions[session_id] = (doc_scope, [])
                self._sessions.move_to_end(session_id)
                return []
            self._sessions.move_to_end(session_id)
            return list(turns)

    def append_exchange(
        self,
        session_id: str,
        user_question: str,
        assistant_answer: str,
        *,
        doc_scope: tuple[str, ...] | None = None,
        exchange_meta: dict[str, Any] | None = None,
    ) -> None:
        """Append a user/assistant exchange to prompt memory and exportable transcript."""
        with self._lock:
            if session_id not in self._sessions and len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            stored_scope, turns = self._sessions.get(session_id, (doc_scope, []))
            if doc_scope is not None and stored_scope is not None and stored_scope != doc_scope:
                turns = []
            turns.append(ConversationTurn(role="user", content=user_question))
            turns.append(ConversationTurn(role="assistant", content=assistant_answer))
            max_items = self._max_turns * 2
            if len(turns) > max_items:
                del turns[: len(turns) - max_items]
            effective_scope = doc_scope if doc_scope is not None else stored_scope
            self._sessions[session_id] = (effective_scope, turns)
            self._sessions.move_to_end(session_id)
            self._append_transcript_locked(session_id, user_question, assistant_answer, exchange_meta)

    def append_transcript_only(
        self,
        session_id: str,
        user_question: str,
        assistant_answer: str,
        *,
        exchange_meta: dict[str, Any] | None = None,
    ) -> None:
        """Record an exchange (e.g. conversational small talk) for transcript export only."""
        with self._lock:
            self._append_transcript_locked(session_id, user_question, assistant_answer, exchange_meta)

    def get_transcript(self, session_id: str) -> list[TranscriptExchange]:
        """Return full recorded transcript exchanges for `session_id`."""
        with self._lock:
            items = self._transcripts.get(session_id)
            if not items:
                return []
            self._transcripts.move_to_end(session_id)
            return list(items)

    def clear_session(self, session_id: str) -> bool:
        """Delete a session's history and transcript; returns True if anything existed."""
        with self._lock:
            removed_ctx = self._sessions.pop(session_id, None) is not None
            removed_tx = self._transcripts.pop(session_id, None) is not None
            return removed_ctx or removed_tx

    def clear_all(self) -> None:
        """Clear all stored sessions and transcripts cleanly (used on knowledge base reset)."""
        with self._lock:
            self._sessions.clear()
            self._transcripts.clear()


def format_transcript_markdown(session_id: str, exchanges: list[TranscriptExchange]) -> str:
    """Render a conversation transcript into clean, shareable Markdown."""
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# AlPrime Document Assistant — Conversation Transcript",
        f"- **Session ID:** `{session_id}`",
        f"- **Exported At:** {exported_at}",
        f"- **Total Turns:** {len(exchanges)}",
        "",
        "---",
        "",
    ]

    for item in exchanges:
        ts = item.timestamp.strftime("%H:%M:%S UTC")
        lines.append(f"### Turn {item.turn} — User ({ts})")
        lines.append(item.question)
        lines.append("")

        status_label = {
            "grounded": "Verified Document Evidence",
            "not_found": item.refusal_reason or "No Supporting Evidence Found",
            "conversational": "Assistant Ready",
        }.get(item.response_type, item.response_type)

        lines.append(f"### Turn {item.turn} — Assistant (`{status_label}`)")
        lines.append(item.answer)
        if item.status_detail and item.response_type == "not_found":
            lines.append(f"\n> *Note: {item.status_detail}*")
        if item.citations:
            lines.append("")
            lines.append("**Cited Sources:**")
            for idx, cit in enumerate(item.citations, start=1):
                name = cit.get("document_name", "unknown")
                page = cit.get("page", 1)
                snippet = " ".join(str(cit.get("snippet", "")).split())
                lines.append(f"- **[{idx}] {name} (Page {page}):** {snippet}")
        lines.extend(["", "---", ""])

    return "\n".join(lines).strip() + "\n"
