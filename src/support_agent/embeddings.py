"""Gemini embedding provider with batching and quota-aware retries."""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

from google import genai
from google.genai import types


EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_BATCH_SIZE = 5
DEFAULT_TPM_LIMIT = 30_000
DEFAULT_TPM_TARGET = 24_000
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF = 2.0


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...
    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass
class _Usage:
    timestamp: float
    tokens: int


class _RateLimiter:
    def __init__(self, target: int):
        if target <= 0:
            raise ValueError("target token rate must be positive")
        self.target = target
        self.window = deque()

    def wait(self, tokens: int):
        if tokens <= 0:
            return
        if tokens > self.target:
            raise ValueError(f"Request requires {tokens} tokens, above TPM target {self.target}")

        while True:
            now = time.monotonic()
            while self.window and self.window[0].timestamp <= now - 60:
                self.window.popleft()
            used = sum(x.tokens for x in self.window)
            if used + tokens <= self.target:
                self.window.append(_Usage(now, tokens))
                return
            time.sleep(max(self.window[0].timestamp + 60 - now, 0.1))


class GeminiEmbeddingProvider:
    def __init__(self, client: genai.Client, model: str = DEFAULT_EMBEDDING_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE, tpm_limit: int = DEFAULT_TPM_LIMIT,
                 tpm_target: int = DEFAULT_TPM_TARGET, max_retries: int = DEFAULT_MAX_RETRIES,
                 base_backoff_seconds: float = DEFAULT_BACKOFF):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if tpm_limit < 1 or not 0 < tpm_target <= tpm_limit:
            raise ValueError("invalid TPM configuration")
        if max_retries < 0 or base_backoff_seconds <= 0:
            raise ValueError("invalid retry configuration")

        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.tpm_limit = tpm_limit
        self.max_retries = max_retries
        self.backoff = base_backoff_seconds
        self.limiter = _RateLimiter(tpm_target)

    @classmethod
    def from_env(cls) -> "GeminiEmbeddingProvider":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY must be configured")

        return cls(
            genai.Client(api_key=key),
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            tpm_limit=int(os.getenv("EMBEDDING_TPM_LIMIT", DEFAULT_TPM_LIMIT)),
            tpm_target=int(os.getenv("EMBEDDING_TPM_TARGET", DEFAULT_TPM_TARGET)),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
            base_backoff_seconds=float(os.getenv("EMBEDDING_BASE_BACKOFF_SECONDS", DEFAULT_BACKOFF)),
        )

    def count_tokens(self, texts: Sequence[str]) -> int:
        if not texts:
            return 0
        result = self.client.models.count_tokens(model=self.model, contents=list(texts))
        total = getattr(result, "total_tokens", None)
        if total is None:
            raise ValueError("Gemini token-count response did not contain total_tokens")
        return int(total)

    @staticmethod
    def _rate_limited(exc: Exception) -> bool:
        return (
            getattr(exc, "code", None) == 429
            or getattr(exc, "status_code", None) == 429
            or ("429" in str(exc) and "RESOURCE_EXHAUSTED" in str(exc))
        )

    def _embed_batch(self, batch: Sequence[str], task_type: str) -> list[list[float]]:
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=list(batch),
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIMENSION,
                    ),
                )
                values = [list(e.values) for e in response.embeddings]
                if len(values) != len(batch):
                    raise ValueError("Gemini returned an unexpected number of embeddings")
                if any(len(v) != EMBEDDING_DIMENSION for v in values):
                    raise ValueError(f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings")
                return values
            except Exception as exc:
                if not self._rate_limited(exc):
                    raise
                last = exc
                if attempt == self.max_retries:
                    break
                delay = min(self.backoff * (2 ** attempt), 60.0)
                time.sleep(delay + random.uniform(0, min(1.0, delay * 0.25)))
        raise last or RuntimeError("embedding request failed")

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            tokens = self.count_tokens(batch)
            if tokens > self.tpm_limit:
                raise ValueError(f"Batch requires {tokens} tokens, exceeding TPM limit {self.tpm_limit}")
            self.limiter.wait(tokens)
            result.extend(self._embed_batch(batch, task_type))
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            return []
        values = self._embed([text], "RETRIEVAL_QUERY")
        return values[0] if values else []