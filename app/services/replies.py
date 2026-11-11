"""Fixed replies that are built from templates, never from the language model.

Because these texts only contain template wording and real filenames, a refusal or
greeting can never introduce information that is not in the documents.
"""

from app.services.prompts import NO_INFO_USER_MESSAGE
from app.services.query_intent import ConversationalIntent

CONVERSATIONAL_STATUS = "Conversational greeting handled without document retrieval."

# Short "why" note shown under a refusal (badge tooltip / grey line in the UI).
REFUSAL_DETAILS: dict[str, str] = {
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


def _describe_documents(filenames: list[str]) -> str:
    """'X.pdf', 'X.pdf and Y.docx', or 'your 5 uploaded documents'."""
    names = list(dict.fromkeys(filenames))
    if not names:
        return "your uploaded documents"
    if len(names) > _MAX_NAMED_DOCS:
        return f"your {len(names)} uploaded documents"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def build_refusal_message(
    reason: str,
    *,
    searched_docs: list[str] | None = None,
    evidence_docs: list[str] | None = None,
) -> str:
    """The fixed "not found" lead sentence, plus what happened and what to do next."""
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


def build_conversational_reply(intent: ConversationalIntent, filenames: list[str]) -> str:
    """Template reply to small talk, mentioning which documents are currently loaded."""
    if intent == "thanks":
        return "You're welcome! Let me know if you have any other questions about your uploaded documents."
    if not filenames:
        return (
            "Hello! I'm the AlPrime Document RAG Assistant. Please upload PDF, DOCX, or TXT "
            "files using the paperclip button below or the Document Vault, and ask me anything "
            "grounded strictly in their contents."
        )
    preview = ", ".join(filenames[:3])
    extra = f" (+{len(filenames) - 3} more)" if len(filenames) > 3 else ""
    if intent == "help":
        return (
            "I answer questions strictly from your uploaded documents with page citations. "
            f"Currently loaded ({len(filenames)}): {preview}{extra}. "
            "Ask a specific question or ask me to summarize them."
        )
    return (
        "Hello! I'm doing well and ready to help. "
        f"You currently have {len(filenames)} document(s) loaded ({preview}{extra}). "
        "Ask me any specific question or ask for a summary. All answers stay grounded in "
        "your uploaded files."
    )
