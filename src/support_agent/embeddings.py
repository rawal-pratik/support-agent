"""Configurable embedding provider for document chunks."""

import os
from collections.abc import Sequence
from typing import Protocol

from google import genai
from google.genai import types

EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
# Gemini API has rate limits; process one text at a time to avoid 429 errors
DEFAULT_BATCH_SIZE = 1


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""
        ...


class GeminiEmbeddingProvider:
    """Google Gemini embedding provider with batching and dimension validation."""

    def __init__(
        self,
        client: genai.Client,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._client = client
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def from_env(cls) -> "GeminiEmbeddingProvider":
        """Create provider from environment variables."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be configured")

        client = genai.Client(api_key=api_key)

        model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))

        return cls(client=client, model=model, batch_size=batch_size)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches, validating dimensions."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])

            # Embed the batch using the new google.genai SDK
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSION
                ),
            )

            # Extract embeddings from response
            batch_embeddings = [emb.values for emb in response.embeddings]

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    f"Embedding response length ({len(batch_embeddings)}) "
                    f"does not match batch size ({len(batch)})"
                )

            for embedding in batch_embeddings:
                if len(embedding) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Expected {EMBEDDING_DIMENSION} dimensions, "
                        f"got {len(embedding)}"
                    )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings
