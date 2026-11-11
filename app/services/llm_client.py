"""LLM clients: Runware (DeepSeek), Google Gemini, Groq (Qwen), any OpenAI-compatible endpoint,
plus a router.

The `RoutingLLMClient` gives the RAG pipeline a single `LLMClient` that:
  * tries providers in order (primary -> fallbacks) when one fails (quota, outage, 5xx);
  * sends questions that need multi-step reasoning to the reasoning provider first
    (DeepSeek on Runware with high reasoning effort), while simple lookups use light reasoning;
  * only falls back mid-stream if the failing provider had not yet emitted any token.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
import re
import time
from typing import Any, TypeVar

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from app.config import Settings
from app.core.exceptions import LLMConfigurationError, LLMProviderError
from app.core.logging import get_logger
from app.core.tracing import current_trace, report_usage, take_usage

logger = get_logger(__name__)
T = TypeVar("T")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_TEMPERATURE = 0.6  # Qwen thinking mode degrades with greedy (0.0) decoding.
_RUNWARE_MIN_THINKING_BUDGET = 1024  # below this explicit max_tokens, DeepSeek thinking is disabled


def _is_daily_quota_error(exc: Exception) -> bool:
    """Detect daily free-tier quota exhaustion where sleeping 1-2s will not help."""
    msg = str(exc).upper()
    return any(marker in msg for marker in ("GENERATEREQUESTSPERDAY", "QUOTA EXCEEDED", "PERDAY"))


def _is_transient_error(exc: Exception) -> bool:
    """Detect short-lived 503/429 spikes safe for quick backoff retry."""
    if _is_daily_quota_error(exc):
        return False
    msg = str(exc).upper()
    return any(code in msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "RATE LIMIT"))


def _is_fallback_eligible(exc: LLMProviderError) -> bool:
    """Determine if a Gemini failure warrants falling back to a sibling Flash model."""
    msg = exc.detail.upper()
    return any(
        code in msg for code in ("503", "429", "404", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "NOT_FOUND")
    )


def strip_reasoning_text(text: str) -> str:
    """Remove any `<think>...</think>` block a reasoning model may leak into its answer."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # An unterminated opening tag means the answer was cut off inside the reasoning.
    if "<think>" in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().index("<think>")]
    return cleaned.strip()


class _ThinkStreamFilter:
    """Incrementally drop `<think>...</think>` spans from a token stream."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, token: str) -> str:
        self._buffer += token
        out: list[str] = []
        while self._buffer:
            lower = self._buffer.lower()
            if self._inside:
                end = lower.find(self._CLOSE)
                if end == -1:
                    # Keep only a possible partial closing tag.
                    self._buffer = self._buffer[-(len(self._CLOSE) - 1) :]
                    break
                self._buffer = self._buffer[end + len(self._CLOSE) :]
                self._inside = False
                continue
            start = lower.find(self._OPEN)
            if start == -1:
                # Hold back a suffix that could be the start of "<think>".
                hold = 0
                for size in range(1, len(self._OPEN)):
                    if lower.endswith(self._OPEN[:size]):
                        hold = size
                emit_upto = len(self._buffer) - hold
                out.append(self._buffer[:emit_upto])
                self._buffer = self._buffer[emit_upto:]
                break
            out.append(self._buffer[:start])
            self._buffer = self._buffer[start + len(self._OPEN) :]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        remainder = "" if self._inside else self._buffer
        self._buffer = ""
        return remainder


async def _call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    provider_label: str,
    max_retries: int = 2,
) -> T:
    """Execute async upstream LLM call with short backoff on transient 503/429 errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except LLMConfigurationError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_transient_error(exc):
                logger.warning("%s request failed: %s", provider_label, exc)
                raise LLMProviderError(
                    f"{provider_label} request failed. Please retry shortly.",
                    detail=str(exc),
                ) from exc
            delay = 1.0 * (attempt + 1)
            logger.warning(
                "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                provider_label,
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise LLMProviderError(f"{provider_label} request failed.", detail=str(last_exc))


class LLMClient(ABC):
    """Abstract interface for non-streaming and streaming chat completion."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable identifier of the configured LLM backend."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True when required credentials are present."""

    @abstractmethod
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> str:
        """Return complete generated response text."""

    @abstractmethod
    def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        """Yield incremental text chunks as the model generates them."""


class GeminiLLMClient(LLMClient):
    """Google GenAI SDK wrapper using Gemini models with deterministic temperature."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.gemini_api_key.strip()
        self._model = settings.gemini_model
        self._default_temp = settings.llm_temperature
        self._default_max_tokens = settings.llm_max_tokens
        self._client: genai.Client | None = None

    @property
    def provider_name(self) -> str:
        return f"gemini ({self._model})"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> genai.Client:
        if not self.is_configured:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is missing. Add it to your .env file or configure another provider."
            )
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_config(
        self,
        *,
        model_name: str,
        system_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        reasoning: bool,
    ) -> genai_types.GenerateContentConfig:
        kwargs: dict[str, object] = {
            "system_instruction": system_prompt,
            "temperature": self._default_temp if temperature is None else temperature,
            "max_output_tokens": self._default_max_tokens if max_tokens is None else max_tokens,
        }
        # Disable hidden reasoning on non-lite Flash models for lower latency on simple lookups.
        lower = model_name.lower()
        if not reasoning and "flash" in lower and "lite" not in lower:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        return genai_types.GenerateContentConfig(**kwargs)

    def _candidate_models(self) -> list[str]:
        """Provide Flash sibling fallback if primary Gemini model hits 429 quota or 503 outage."""
        models_list = [self._model]
        lower = self._model.lower()
        if "flash-lite" in lower and self._model != "gemini-3.5-flash":
            models_list.append("gemini-3.5-flash")
        elif "flash" in lower and self._model != "gemini-3.5-flash-lite":
            models_list.append("gemini-3.5-flash-lite")
        return models_list

    async def _open(
        self,
        call: Callable[[str, genai_types.GenerateContentConfig], Awaitable[Any]],
        *,
        system_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        reasoning: bool,
        label: str,
    ) -> Any:
        """Run `call` against the primary Gemini model, then its Flash sibling on quota/outage."""
        candidates = self._candidate_models()
        for idx, model_name in enumerate(candidates):
            config = self._build_config(
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            try:
                return await _call_with_retry(
                    lambda m=model_name, cfg=config: call(m, cfg),
                    provider_label=label,
                )
            except LLMProviderError as exc:
                if idx == len(candidates) - 1 or not _is_fallback_eligible(exc):
                    raise
                logger.warning(
                    "Gemini model '%s' failed; falling back to '%s'", model_name, candidates[idx + 1]
                )
        raise LLMProviderError(f"{label} request failed.")

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> str:
        client = self._get_client()
        response = await self._open(
            lambda m, cfg: client.aio.models.generate_content(model=m, contents=user_prompt, config=cfg),
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            label="Gemini API",
        )
        report_usage(getattr(response, "usage_metadata", None))
        text = strip_reasoning_text((response.text if response else "") or "")
        if not text:
            raise LLMProviderError("Gemini API returned an empty response.")
        return text

    async def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        response_stream = await self._open(
            lambda m, cfg: client.aio.models.generate_content_stream(
                model=m, contents=user_prompt, config=cfg
            ),
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            label="Gemini streaming",
        )
        try:
            async for chunk in response_stream:
                if getattr(chunk, "usage_metadata", None) is not None:
                    report_usage(chunk.usage_metadata)
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.warning("Gemini streaming interrupted: %s", exc)
            raise LLMProviderError("Gemini streaming was interrupted.", detail=str(exc)) from exc


class OpenAICompatibleLLMClient(LLMClient):
    """OpenAI SDK client for any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, ...)."""

    _label = "openai_compatible"
    # Ask for token usage on streamed responses. Off for arbitrary endpoints (some local
    # servers reject `stream_options`); enabled for providers verified to support it.
    _stream_usage = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        reasoning_max_tokens: int | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.strip()
        self._model = model
        self._default_temp = temperature
        self._default_max_tokens = max_tokens
        self._reasoning_max_tokens = reasoning_max_tokens or max_tokens
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleLLMClient":
        return cls(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            reasoning_max_tokens=settings.llm_reasoning_max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return f"{self._label} ({self._model})"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> AsyncOpenAI:
        if not self.is_configured:
            raise LLMConfigurationError(
                f"API key is missing for the '{self._label}' provider. Set it in .env."
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url or None,
                timeout=self._timeout,
            )
        return self._client

    def _extra_body(self, reasoning: bool) -> dict[str, Any]:
        """Provider-specific request fields (none for generic endpoints)."""
        return {}

    def _request_kwargs(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        reasoning: bool,
        include_extras: bool,
    ) -> dict[str, Any]:
        if temperature is None:
            temperature = max(self._default_temp, _REASONING_TEMPERATURE) if reasoning else self._default_temp
        if max_tokens is None:
            max_tokens = self._reasoning_max_tokens if reasoning else self._default_max_tokens
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extras = self._extra_body(reasoning) if include_extras else {}
        if extras:
            kwargs["extra_body"] = extras
        return kwargs

    def _stream_kwargs(self, stream: bool) -> dict[str, Any]:
        if stream and self._stream_usage:
            return {"stream": True, "stream_options": {"include_usage": True}}
        return {"stream": stream}

    async def _create(self, *, stream: bool, **request: Any) -> Any:
        """Call the endpoint; if it rejects provider-specific reasoning fields, retry without them."""
        client = self._get_client()
        label = f"{self._label} {'streaming' if stream else 'API'}"
        try:
            return await _call_with_retry(
                lambda: client.chat.completions.create(
                    **self._request_kwargs(**request, include_extras=True), **self._stream_kwargs(stream)
                ),
                provider_label=label,
            )
        except LLMProviderError as exc:
            if not self._extra_body(request["reasoning"]) or "REASONING" not in exc.detail.upper():
                raise
            logger.warning("%s rejected reasoning parameters; retrying without them.", label)
            return await _call_with_retry(
                lambda: client.chat.completions.create(
                    **self._request_kwargs(**request, include_extras=False), **self._stream_kwargs(stream)
                ),
                provider_label=label,
            )

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> str:
        completion = await self._create(
            stream=False,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
        report_usage(getattr(completion, "usage", None))
        content = (
            completion.choices[0].message.content
            if completion.choices and completion.choices[0].message
            else None
        )
        text = strip_reasoning_text(content or "")
        if not text:
            raise LLMProviderError(f"{self._label} API returned an empty response.")
        return text

    async def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        response_stream = await self._create(
            stream=True,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
        think_filter = _ThinkStreamFilter()
        try:
            async for chunk in response_stream:
                if getattr(chunk, "usage", None) is not None:
                    report_usage(chunk.usage)
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    visible = think_filter.feed(chunk.choices[0].delta.content)
                    if visible:
                        yield visible
            tail = think_filter.flush()
            if tail:
                yield tail
        except Exception as exc:
            logger.warning("%s streaming interrupted: %s", self._label, exc)
            raise LLMProviderError(
                f"{self._label} streaming was interrupted.", detail=str(exc)
            ) from exc


class GroqLLMClient(OpenAICompatibleLLMClient):
    """Groq OpenAI-compatible endpoint running Qwen with switchable thinking mode."""

    _label = "groq"
    _stream_usage = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            reasoning_max_tokens=settings.llm_reasoning_max_tokens,
        )
        self._reasoning_effort = settings.groq_reasoning_effort.strip()
        self._non_reasoning_effort = settings.groq_non_reasoning_effort.strip()

    def _extra_body(self, reasoning: bool) -> dict[str, Any]:
        effort = self._reasoning_effort if reasoning else self._non_reasoning_effort
        if not effort:
            return {}
        body: dict[str, Any] = {"reasoning_effort": effort}
        if effort != "none":
            # Keep the chain-of-thought out of the answer text (sentinel + citation safety).
            body["reasoning_format"] = "hidden"
        return body


class RunwareLLMClient(OpenAICompatibleLLMClient):
    """Runware's OpenAI-compatible endpoint running DeepSeek V4.1 Flash.

    DeepSeek returns its chain-of-thought in a separate `reasoning_content` field, so the
    answer text (and the NO_RELEVANT_INFO_FOUND sentinel) stays clean. `reasoning_effort`
    scales the thinking depth per question type.
    """

    _label = "runware"
    _stream_usage = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            api_key=settings.runware_api_key,
            base_url=settings.runware_base_url,
            model=settings.runware_model,
            temperature=settings.llm_temperature,
            max_tokens=max(settings.llm_max_tokens, settings.runware_max_tokens),
            timeout=settings.llm_timeout_seconds,
            reasoning_max_tokens=max(settings.llm_reasoning_max_tokens, settings.runware_max_tokens),
        )
        self._reasoning_effort = settings.runware_reasoning_effort.strip()
        self._non_reasoning_effort = settings.runware_non_reasoning_effort.strip()

    def _extra_body(self, reasoning: bool) -> dict[str, Any]:
        effort = self._reasoning_effort if reasoning else self._non_reasoning_effort
        return {"reasoning_effort": effort} if effort else {}

    def _request_kwargs(self, *, max_tokens: int | None, reasoning: bool, **rest: Any) -> dict[str, Any]:
        kwargs = super()._request_kwargs(max_tokens=max_tokens, reasoning=reasoning, **rest)
        # DeepSeek's thinking tokens count against max_tokens. A caller that caps output at a
        # small budget (e.g. the 320-token follow-up rewrite) would otherwise get the whole
        # budget spent on thinking and an empty answer, forcing a slow failover.
        if (
            not reasoning
            and max_tokens is not None
            and max_tokens < _RUNWARE_MIN_THINKING_BUDGET
            and "extra_body" in kwargs
        ):
            kwargs["extra_body"] = {**kwargs["extra_body"], "reasoning_effort": "none"}
        return kwargs


class RoutingLLMClient(LLMClient):
    """Ordered provider chain with automatic failover and reasoning-aware routing."""

    def __init__(
        self,
        providers: list[tuple[str, LLMClient]],
        *,
        reasoning_provider: str | None = None,
        first_token_timeout: float | None = None,
    ) -> None:
        if not providers:
            raise ValueError("RoutingLLMClient needs at least one provider.")
        self._providers = providers
        self._reasoning_provider = reasoning_provider
        self._first_token_timeout = first_token_timeout

    @property
    def provider_name(self) -> str:
        names = [client.provider_name for _, client in self._providers if client.is_configured]
        if not names:
            names = [self._providers[0][1].provider_name]
        label = " -> ".join(names)
        if self._reasoning_provider and self._reasoning_client() is not None:
            label += f" | reasoning: {self._reasoning_provider}"
        return label

    @property
    def is_configured(self) -> bool:
        return any(client.is_configured for _, client in self._providers)

    def _reasoning_client(self) -> LLMClient | None:
        for name, client in self._providers:
            if name == self._reasoning_provider and client.is_configured:
                return client
        return None

    def _chain(self, reasoning: bool) -> list[tuple[str, LLMClient]]:
        configured = [(n, c) for n, c in self._providers if c.is_configured]
        if not configured:
            raise LLMConfigurationError(
                "No LLM provider is configured. Set RUNWARE_API_KEY, GEMINI_API_KEY and/or GROQ_API_KEY in .env."
            )
        if reasoning and self._reasoning_provider:
            preferred = [(n, c) for n, c in configured if n == self._reasoning_provider]
            others = [(n, c) for n, c in configured if n != self._reasoning_provider]
            return preferred + others
        return configured

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> str:
        chain = self._chain(reasoning)
        last_exc: Exception | None = None
        for idx, (name, client) in enumerate(chain):
            begin = time.perf_counter()
            take_usage()
            try:
                result = await client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                )
                logger.info("LLM answer served by %s (reasoning=%s)", name, reasoning)
                _trace_call(name, client, "generate", reasoning, begin, failovers=idx)
                return result
            except (LLMProviderError, LLMConfigurationError) as exc:
                last_exc = exc
                _trace_call(name, client, "generate", reasoning, begin, failovers=idx, error=exc)
                if idx < len(chain) - 1:
                    logger.warning("LLM provider '%s' failed; failing over to '%s'", name, chain[idx + 1][0])
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool = False,
    ) -> AsyncIterator[str]:
        chain = self._chain(reasoning)
        last_exc: Exception | None = None
        for idx, (name, client) in enumerate(chain):
            emitted = False
            begin = time.perf_counter()
            take_usage()
            tokens = client.stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            try:
                first = await self._first_token(name, tokens)
                if first is None:
                    # The stream ended without any answer text, e.g. a thinking model spent its
                    # whole completion budget on reasoning. Treat it like `generate()` treats an
                    # empty response: a provider failure, so the next provider answers instead of
                    # the empty reply being mistaken for "no relevant information".
                    raise LLMProviderError(
                        f"{name} returned an empty streamed response.",
                        detail="stream ended before any answer text",
                    )
                emitted = True
                yield first
                async for token in tokens:
                    yield token
                logger.info("LLM stream served by %s (reasoning=%s)", name, reasoning)
                _trace_call(name, client, "stream", reasoning, begin, failovers=idx)
                return
            except (LLMProviderError, LLMConfigurationError) as exc:
                _trace_call(name, client, "stream", reasoning, begin, failovers=idx, error=exc)
                # Once tokens reached the client we cannot splice in another model's answer.
                if emitted:
                    raise
                last_exc = exc
                if idx < len(chain) - 1:
                    logger.warning(
                        "LLM stream provider '%s' failed; failing over to '%s'", name, chain[idx + 1][0]
                    )
        assert last_exc is not None
        raise last_exc

    async def _first_token(self, name: str, tokens: AsyncIterator[str]) -> str | None:
        """Wait for a provider's first answer token, giving up after the stall timeout.

        Thinking models stream their reasoning separately, so the HTTP connection stays busy
        (no read timeout fires) while the user sees nothing. A provider that has not produced
        any answer text in time is closed and treated as failed, so the next one can answer.
        """
        try:
            return await asyncio.wait_for(anext(tokens), timeout=self._first_token_timeout)
        except StopAsyncIteration:
            return None
        except asyncio.TimeoutError as exc:
            await _aclose_quietly(tokens)
            logger.warning(
                "LLM stream provider '%s' sent no answer text within %.0fs",
                name,
                self._first_token_timeout,
            )
            raise LLMProviderError(
                f"{name} did not start answering in time.",
                detail=f"first-token timeout {self._first_token_timeout}s",
            ) from exc


async def _aclose_quietly(tokens: AsyncIterator[str]) -> None:
    aclose = getattr(tokens, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def _trace_call(
    name: str,
    client: LLMClient,
    kind: str,
    reasoning: bool,
    begin: float,
    *,
    failovers: int,
    error: Exception | None = None,
) -> None:
    """Attach one provider attempt (latency, tokens, failover position) to the active trace."""
    usage = take_usage()
    trace = current_trace()
    if trace is None:
        return
    trace.record_llm_call(
        provider=name,
        model=client.provider_name,
        kind=kind,
        reasoning=reasoning,
        ms=int(round((time.perf_counter() - begin) * 1000)),
        failover_index=failovers or None,
        error=type(error).__name__ if error else None,
        **usage,
    )


def _build_single_client(name: str, settings: Settings) -> LLMClient:
    if name == "runware":
        return RunwareLLMClient(settings)
    if name == "gemini":
        return GeminiLLMClient(settings)
    if name == "groq":
        return GroqLLMClient(settings)
    if name == "openai_compatible":
        return OpenAICompatibleLLMClient.from_settings(settings)
    raise ValueError(f"Unknown LLM provider '{name}'. Use runware, gemini, groq, or openai_compatible.")


def build_llm_client(settings: Settings) -> LLMClient:
    """Build the routed provider chain: primary first, then configured fallbacks."""
    order = [settings.llm_provider, *settings.fallback_provider_list]
    providers = [(name, _build_single_client(name, settings)) for name in order]
    reasoning_provider = settings.reasoning_provider if settings.enable_reasoning_routing else None
    if reasoning_provider and reasoning_provider not in order:
        providers.append((reasoning_provider, _build_single_client(reasoning_provider, settings)))
    return RoutingLLMClient(
        providers,
        reasoning_provider=reasoning_provider,
        first_token_timeout=settings.llm_first_token_timeout_seconds,
    )
