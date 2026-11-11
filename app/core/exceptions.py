"""Domain-specific exception hierarchy mapped to clean HTTP status codes."""


class AppError(Exception):
    """Base application error carrying an HTTP status code and machine-readable code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_code: str = "internal_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class UnsupportedFormatError(AppError):
    """Raised when an uploaded file extension is not PDF, DOCX, or TXT."""

    def __init__(self, filename: str, supported: set[str]) -> None:
        ext_list = ", ".join(sorted(supported))
        super().__init__(
            f"Unsupported file format for '{filename}'. Allowed formats: {ext_list}.",
            status_code=400,
            error_code="unsupported_format",
        )


class FileTooLargeError(AppError):
    """Raised when an uploaded file exceeds the configured upload size limit."""

    def __init__(self, filename: str, max_mb: int) -> None:
        super().__init__(
            f"File '{filename}' exceeds the maximum allowed size of {max_mb} MB.",
            status_code=413,
            error_code="file_too_large",
        )


class TooManyFilesError(AppError):
    """Raised when one upload request contains more files than allowed."""

    def __init__(self, max_files: int) -> None:
        super().__init__(
            f"Too many files in one request. The maximum is {max_files} files per upload.",
            status_code=413,
            error_code="too_many_files",
        )


class EmptyDocumentError(AppError):
    """Raised when a document is empty or yields no extractable text."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=422,
            error_code="empty_document",
        )


class DocumentParsingError(AppError):
    """Raised when a document is corrupt, encrypted, or unreadable."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=422,
            error_code="document_parsing_error",
        )


class DocumentNotFoundError(AppError):
    """Raised when querying or deleting an unknown document ID."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            f"Document '{doc_id}' was not found.",
            status_code=404,
            error_code="document_not_found",
        )


class VectorStoreError(AppError):
    """Raised on Qdrant or FastEmbed runtime failures."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=500,
            error_code="vector_store_error",
        )


class LLMConfigurationError(AppError):
    """Raised when no valid LLM API key or provider configuration is set."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=503,
            error_code="llm_not_configured",
        )


class LLMProviderError(AppError):
    """Raised when the upstream LLM API call fails or times out.

    `message` is safe to return to API clients; `detail` holds the raw upstream error
    for server-side logging and fallback decisions and is never sent to clients.
    """

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(
            message,
            status_code=502,
            error_code="llm_provider_error",
        )
        self.detail = detail or message
