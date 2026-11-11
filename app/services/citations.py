"""Post-processing of model output: refusal detection and citation markers."""

import html
import re

from app.services.prompts import NO_INFO_SENTINEL
from app.services.vector_store import RetrievedChunk

_CITATION_MARKER_RE = re.compile(r"\[\s*(?:\d+(?:\s*,\s*\d+)*|Source:)", re.IGNORECASE)
_SOURCE_TAG_RE = re.compile(r"\[Source:\s*(?P<name>[^|\]]+?)\s*\|\s*Page\s*(?P<page>\d+)\s*\]", re.IGNORECASE)
_INDEX_TAG_RE = re.compile(r"\[(?P<nums>\d+(?:\s*,\s*\d+)*)\](?!\()")
_CODE_WRAPPED_MARKER_RE = re.compile(r"`((?:\[\s*\d+(?:\s*,\s*\d+)*\s*\])+)`")


def has_citation_marker(text: str) -> bool:
    return bool(_CITATION_MARKER_RE.search(text))


def sentinel_holdback(text: str) -> int:
    """Length of the longest suffix of `text` that could be the start of the sentinel.

    While streaming, that many trailing characters are held back so a half-received
    sentinel never reaches the user.
    """
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
    return rest if has_citation_marker(rest) else None


def unwrap_code_markers(answer: str) -> str:
    """Remove inline-code backticks some models put around citation markers (`[1]` -> [1])."""
    return _CODE_WRAPPED_MARKER_RE.sub(lambda m: m.group(1), answer)


def _parse_indices(match: re.Match[str], limit: int) -> list[int] | None:
    """Marker numbers of an `[n, m]` match, or None if any is out of range."""
    nums = [int(part) for part in match.group("nums").split(",")]
    return nums if all(1 <= n <= limit for n in nums) else None


def normalize_cited_chunks(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """Keep only the chunks the answer cites, renumbering `[k]` markers in citation order.

    Accepts compact `[1]` markers and the longer `[Source: <file> | Page <n>]` form. An answer
    with no recognisable markers keeps every chunk.
    """
    answer = unwrap_code_markers(answer)

    cited_pages = {
        (html.unescape(m.group("name")).strip().lower(), int(m.group("page")))
        for m in _SOURCE_TAG_RE.finditer(answer)
    }
    if cited_pages:
        kept = [c for c in chunks if (c.document_name.strip().lower(), c.page) in cited_pages]
        return answer, (kept or chunks)

    limit = len(chunks)
    cited_order: list[int] = []
    for match in _INDEX_TAG_RE.finditer(answer):
        for n in _parse_indices(match, limit) or []:
            if n not in cited_order:
                cited_order.append(n)
    if not cited_order:
        return answer, chunks

    renumber = {old: new for new, old in enumerate(cited_order, start=1)}

    def _renumber(match: re.Match[str]) -> str:
        nums = _parse_indices(match, limit)
        if nums is None:
            return match.group(0)
        return "".join(f"[{n}]" for n in dict.fromkeys(renumber[n] for n in nums))

    return _INDEX_TAG_RE.sub(_renumber, answer), [chunks[n - 1] for n in cited_order]
