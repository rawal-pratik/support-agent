"""Configurable embedding provider for document chunks."""

from __future__ import annotations

import os
import random
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import types


EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

# Maximum number of texts sent in a single Gemini request.
DEFAULT_BATCH_SIZE = 5

# Gemini's currently configured TPM quota.
DEFAULT_TPM_LIMIT = 30_000

# Operate below the hard quota to leave a safety margin.
DEFAULT_TPM_TARGET = 24_000

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_BACKOFF_SECONDS = 2.0


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""
        ...


@dataclass(frozen=True)
class _TokenRequest:
    """Record of a Gemini request and its token usage."""

    timestamp: float
    tokens: int


class _TokenRateLimiter:
    """Rolling-window limiter for Gemini token usage."""

    WINDOW_SECONDS = 60.0

    def __init__(self, target_tokens_per_minute: int):
        if target_tokens_per_minute <= 0:
            raise ValueError("target_tokens_per_minute must be positive")

        self._target = target_tokens_per_minute
        self._requests: deque[_TokenRequest] = deque()

    def _remove_expired(self, now: float) -> None:
        cutoff = now - self.WINDOW_SECONDS

        while self._requests and self._requests[0].timestamp <= cutoff:
            self._requests.popleft()

    def _current_usage(self, now: float) -> int:
        self._remove_expired(now)
        return sum(request.tokens for request in self._requests)

    def wait_for_capacity(self, tokens: int) -> None:
        """Wait until a request fits within the rolling token budget."""

        if tokens <= 0:
            return

        if tokens > self._target:
            raise ValueError(
                f"Single embedding request requires approximately {tokens} "
                f"tokens, which exceeds the configured TPM target of "
                f"{self._target}"
            )

        while True:
            now = time.monotonic()
            current_usage = self._current_usage(now)

            if current_usage + tokens <= self._target:
                self._requests.append(
                    _TokenRequest(timestamp=now, tokens=tokens)
                )
                return

            oldest = self._requests[0]
            wait_seconds = (
                oldest.timestamp + self.WINDOW_SECONDS - now
            )

            time.sleep(max(wait_seconds, 0.1))


class GeminiEmbeddingProvider:
    """Google Gemini embedding provider with quota-aware batching."""

    def __init__(
        self,
        client: genai.Client,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        tpm_limit: int = DEFAULT_TPM_LIMIT,
        tpm_target: int = DEFAULT_TPM_TARGET,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if tpm_limit <= 0:
            raise ValueError("tpm_limit must be positive")

        if tpm_target <= 0 or tpm_target > tpm_limit:
            raise ValueError(
                "tpm_target must be positive and cannot exceed tpm_limit"
            )

        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be positive")

        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._tpm_limit = tpm_limit
        self._tpm_target = tpm_target
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds

        self._rate_limiter = _TokenRateLimiter(
            target_tokens_per_minute=tpm_target
        )

    @classmethod
    def from_env(cls) -> "GeminiEmbeddingProvider":
        """Create provider from environment variables."""

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be configured")

        client = genai.Client(api_key=api_key)

        model = os.getenv(
            "EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )

        batch_size = int(
            os.getenv(
                "EMBEDDING_BATCH_SIZE",
                str(DEFAULT_BATCH_SIZE),
            )
        )

        tpm_limit = int(
            os.getenv(
                "EMBEDDING_TPM_LIMIT",
                str(DEFAULT_TPM_LIMIT),
            )
        )

        tpm_target = int(
            os.getenv(
                "EMBEDDING_TPM_TARGET",
                str(DEFAULT_TPM_TARGET),
            )
        )

        max_retries = int(
            os.getenv(
                "EMBEDDING_MAX_RETRIES",
                str(DEFAULT_MAX_RETRIES),
            )
        )

        base_backoff_seconds = float(
            os.getenv(
                "EMBEDDING_BASE_BACKOFF_SECONDS",
                str(DEFAULT_BASE_BACKOFF_SECONDS),
            )
        )

        return cls(
            client=client,
            model=model,
            batch_size=batch_size,
            tpm_limit=tpm_limit,
            tpm_target=tpm_target,
            max_retries=max_retries,
            base_backoff_seconds=base_backoff_seconds,
        )

    def count_tokens(self, texts: Sequence[str]) -> int:
        """Return Gemini's token count for the supplied input texts."""

        if not texts:
            return 0

        response = self._client.models.count_tokens(
            model=self._model,
            contents=list(texts),
        )

        total_tokens = getattr(response, "total_tokens", None)

        if total_tokens is None:
            raise ValueError(
                "Gemini token-count response did not contain total_tokens"
            )

        return int(total_tokens)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Return whether an exception represents a Gemini 429 response."""

        code = getattr(error, "code", None)

        if code == 429:
            return True

        status_code = getattr(error, "status_code", None)

        if status_code == 429:
            return True

        return "429" in str(error) and "RESOURCE_EXHAUSTED" in str(error)

    def _retry_delay(self, attempt: int) -> float:
        """Calculate bounded exponential backoff with jitter."""

        exponential = self._base_backoff_seconds * (2**attempt)

        # Prevent an unexpectedly large retry delay.
        capped = min(exponential, 60.0)

        # Small random component prevents synchronized retries.
        jitter = random.uniform(0, min(1.0, capped * 0.25))

        return capped + jitter

    def _embed_batch(
        self,
        batch: Sequence[str],
    ) -> list[list[float]]:
        """Embed one batch with bounded 429 retries."""

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=list(batch),
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIMENSION,
                    ),
                )

                batch_embeddings = [
                    list(embedding.values)
                    for embedding in response.embeddings
                ]

                if len(batch_embeddings) != len(batch):
                    raise ValueError(
                        f"Embedding response length "
                        f"({len(batch_embeddings)}) does not match "
                        f"batch size ({len(batch)})"
                    )

                for embedding in batch_embeddings:
                    if len(embedding) != EMBEDDING_DIMENSION:
                        raise ValueError(
                            f"Expected {EMBEDDING_DIMENSION} dimensions, "
                            f"got {len(embedding)}"
                        )

                return batch_embeddings

            except Exception as error:
                if not self._is_rate_limit_error(error):
                    raise

                last_error = error

                if attempt >= self._max_retries:
                    break

                delay = self._retry_delay(attempt)
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in quota-aware batches."""

        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])

            # Count the actual Gemini input tokens before making the
            # embedding request so the rolling TPM limiter can schedule it.
            token_count = self.count_tokens(batch)

            if token_count > self._tpm_limit:
                raise ValueError(
                    f"Batch requires {token_count} tokens, exceeding "
                    f"the configured Gemini TPM limit of {self._tpm_limit}"
                )

            self._rate_limiter.wait_for_capacity(token_count)

            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings