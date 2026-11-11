"""Application configuration backed by environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["runware", "gemini", "groq", "openai_compatible"]

DEFAULT_API_KEY = "alprime-dev-key"


class Settings(BaseSettings):
    """Centralized typed settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "AlPrime Document RAG Assistant"
    app_version: str = "1.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    # Comma-separated list of allowed browser origins ("*" = any origin, no credentials).
    cors_allow_origins: str = "*"

    # LLM provider routing
    # Primary provider for normal questions.
    llm_provider: LLMProviderName = "runware"
    # Comma-separated providers tried, in order, when the primary fails (quota, outage, 5xx).
    # Providers without an API key are skipped automatically.
    llm_fallback_providers: str = "gemini,groq"
    # Route analytical questions (comparisons, calculations, "why" questions, multi-part)
    # to the reasoning provider first.
    enable_reasoning_routing: bool = True
    reasoning_provider: LLMProviderName = "runware"
    llm_reasoning_max_tokens: int = Field(default=4096, ge=256, le=32768)

    # Runware (OpenAI-compatible) running DeepSeek V4.1 Flash: primary + reasoning model.
    runware_api_key: str = ""
    runware_base_url: str = "https://api.runware.ai/v1"
    runware_model: str = Field(
        default="deepseek:v4.1@flash-1017",
        validation_alias=AliasChoices(
            "runware_model", "RUNWARE_MODEL", "RUNWARE_LLM_MODEL", "RUNWARE_LLM_MODEL_API_KEY"
        ),
    )
    # DeepSeek's thinking depth: light for lookups, deep for analytical questions ("none" disables).
    runware_reasoning_effort: str = "high"
    runware_non_reasoning_effort: str = "low"
    # Reasoning tokens count against the completion budget, so DeepSeek gets more headroom.
    # 2048 proved too small: a six-document summary once spent all 2048 tokens on thinking
    # and returned no answer text (the router now fails over on that, but it costs ~12 s).
    runware_max_tokens: int = Field(default=4096, ge=256, le=32768)

    # Google Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # Groq (OpenAI-compatible) running Qwen 27B
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "qwen/qwen3.8-27b"
    # Qwen on Groq supports reasoning_effort "default" (thinking) and "none" (instruct mode).
    groq_reasoning_effort: str = "default"
    groq_non_reasoning_effort: str = "none"

    # Any other OpenAI-compatible endpoint (OpenAI, OpenRouter, Runware, Ollama, ...)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)
    llm_timeout_seconds: int = Field(default=45, ge=5, le=300)
    # A streaming provider that sends no answer text within this window (e.g. stuck in a long
    # thinking phase or a stalled connection) is abandoned and the next provider is tried.
    llm_first_token_timeout_seconds: float = Field(default=20.0, ge=3.0, le=300.0)

    # Qdrant Vector Store
    qdrant_path: str = "./data/qdrant"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "alprime_documents"

    # Local ONNX Embeddings & Reranker (FastEmbed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    enable_reranker: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # RAG & Anti-Hallucination Guardrails
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    retrieval_top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_k: int = Field(default=6, ge=1, le=20)
    min_vector_score: float = Field(default=0.45, ge=0.0, le=1.0)
    # Cross-encoder gate. Calibrated on the sample corpus: off-topic noise scores ~-11,
    # genuine answers score as low as ~-5 (e.g. "overtime hourly rate" = -3.99).
    min_rerank_score: float = -6.0
    # Score each candidate by max(whole chunk, best 2-sentence window). A chunk that mixes
    # topics (e.g. a fee table followed by the payment-terms sentence) otherwise scores at
    # noise level (~-11) even though one sentence answers the question exactly.
    rerank_window_scoring: bool = True
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    max_upload_files: int = Field(default=10, ge=1, le=100)

    # API Authentication (Optional Bonus)
    auth_enabled: bool = False
    api_key: str = DEFAULT_API_KEY

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than CHUNK_SIZE ({self.chunk_size})."
            )
        if self.auth_enabled and (not self.api_key.strip() or self.api_key == DEFAULT_API_KEY):
            raise ValueError(
                "AUTH_ENABLED=true requires a non-default API_KEY. Set a strong random API_KEY in .env."
            )
        return self

    @property
    def fallback_provider_list(self) -> list[str]:
        """Parsed, de-duplicated fallback providers (excluding the primary)."""
        seen: list[str] = []
        for raw in self.llm_fallback_providers.split(","):
            name = raw.strip().lower()
            if name and name != self.llm_provider and name not in seen:
                seen.append(name)
        return seen

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed CORS origins list."""
        origins = [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]
        return origins or ["*"]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
