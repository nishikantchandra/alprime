"""Rule-based question classification used to route a query before retrieval.

These checks are deliberately simple and deterministic. They run before any model
call, so they must be cheap, predictable and easy to test.
"""

import re
from typing import Literal

from app.schemas.document import DocumentInfo

ConversationalIntent = Literal["greeting", "thanks", "help"]

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_PUNCT_STRIP_RE = re.compile(r"[^\w\s']")


def tokenize_words(text: str) -> set[str]:
    """Lower-cased word tokens of `text` (used to ignore filenames in overview detection)."""
    return set(_WORD_RE.findall(text.lower()))


# --- Small talk -----------------------------------------------------------------------

_GREETING_PHRASES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hiya",
        "howdy",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are u",
        "how r u",
        "hows it going",
        "how's it going",
        "how is it going",
        "whats up",
        "what's up",
        "what is up",
        "hello how are you",
        "hi how are you",
        "hey how are you",
        "hi there how are you",
        "hello there how are you",
        "good morning how are you",
        "hi good morning",
        "hello good morning",
    }
)
_THANKS_PHRASES = frozenset(
    {
        "thanks",
        "thank you",
        "thank you very much",
        "thanks a lot",
        "thx",
        "ty",
        "ok",
        "okay",
        "got it",
        "great",
        "cool",
        "awesome",
        "nice",
        "bye",
        "goodbye",
    }
)
_HELP_PHRASES = frozenset({"who are you", "what are you", "what can you do", "help", "how do you work"})


def classify_conversational_intent(question: str) -> ConversationalIntent | None:
    """Detect pure small talk (greetings, thanks, "what can you do").

    Only exact phrase matches count, so "hi, what was Q3 revenue?" is still a real question.
    """
    normalized = " ".join(_PUNCT_STRIP_RE.sub(" ", question.lower()).split())
    if normalized in _GREETING_PHRASES:
        return "greeting"
    if normalized in _THANKS_PHRASES:
        return "thanks"
    if normalized in _HELP_PHRASES:
        return "help"
    return None


# --- Whole-document overview requests -------------------------------------------------

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
# question goes through normal semantic search instead.
_OVERVIEW_FILLER_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "our",
        "your",
        "all",
        "each",
        "every",
        "both",
        "any",
        "some",
        "what",
        "whats",
        "what's",
        "is",
        "are",
        "was",
        "were",
        "be",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "will",
        "should",
        "shall",
        "may",
        "might",
        "you",
        "me",
        "i",
        "we",
        "us",
        "please",
        "pls",
        "kindly",
        "now",
        "just",
        "quickly",
        "briefly",
        "short",
        "brief",
        "main",
        "key",
        "overall",
        "general",
        "give",
        "tell",
        "show",
        "provide",
        "write",
        "make",
        "get",
        "let",
        "help",
        "explain",
        "describe",
        "summarise",
        "summarize",
        "summary",
        "overview",
        "synopsis",
        "outline",
        "gist",
        "about",
        "of",
        "in",
        "on",
        "for",
        "to",
        "from",
        "with",
        "and",
        "or",
        "into",
        "within",
        "inside",
        "cover",
        "covers",
        "covered",
        "contain",
        "contains",
        "contained",
        "read",
        "know",
        "say",
        "says",
        "said",
        "talk",
        "talks",
        "discuss",
        "discusses",
        "mention",
        "mentions",
        "mentioned",
        "written",
        "list",
        "listed",
        "state",
        "stated",
        "everything",
        "detail",
        "details",
        "highlight",
        "highlights",
        "takeaway",
        "takeaways",
        "purpose",
        "scope",
        "asked",
        "asking",
        "mean",
        "means",
        "stuff",
        "things",
        "doc",
        "docs",
        "document",
        "documents",
        "pdf",
        "pdfs",
        "file",
        "files",
        "report",
        "reports",
        "text",
        "page",
        "pages",
        "uploaded",
        "selected",
        "attached",
        "provided",
        "given",
        "current",
        "whole",
        "entire",
        "content",
        "contents",
        "topic",
        "topics",
        "point",
        "points",
        "idea",
        "ideas",
        "high",
        "level",
        "high-level",
        "new",
        "newly",
        "latest",
        "newest",
        "recent",
        "most",
        "last",
        "added",
        "second",
        "first",
        "another",
        "next",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "few",
        "couple",
        "several",
        "bullet",
        "bullets",
        "bulleted",
        "sentence",
        "sentences",
        "paragraph",
        "paragraphs",
        "line",
        "lines",
        "word",
        "words",
        "numbered",
        "format",
        "table",
        "form",
        "way",
        "style",
        "using",
        "use",
        "concise",
        "short-form",
    ]
)


def is_document_overview_query(question: str, *, ignored_words: set[str] | None = None) -> bool:
    """Detect broad whole-document prompts ("summarize this report", "what is this doc about").

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
    topic_words = [w for w in _WORD_RE.findall(cleaned.lower()) if w not in skip and len(w) > 1]
    return not topic_words


# --- "The latest upload" references ---------------------------------------------------

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


def resolve_recency_doc_ids(question: str, docs: list[DocumentInfo]) -> list[str] | None:
    """Narrow retrieval to the newest upload when the question refers to "the latest file".

    `docs` must be ordered newest first. Returns None when no narrowing applies.
    """
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


# --- Questions that need deeper reasoning ---------------------------------------------

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


def needs_reasoning(question: str) -> bool:
    """Comparisons, calculations, causal or multi-part questions get the reasoning model."""
    if _REASONING_RE.search(question):
        return True
    return question.count("?") > 1 or len(_QUESTION_WORD_RE.findall(question)) >= 3
