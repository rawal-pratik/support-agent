"""Tests for the embedding provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from support_agent.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    GeminiEmbeddingProvider,
)


def make_embedding_response(embeddings: list[list[float]]) -> SimpleNamespace:
    """Create a fake Gemini embedding response."""
    embedding_objects = [
        SimpleNamespace(values=emb) for emb in embeddings
    ]
    return SimpleNamespace(embeddings=embedding_objects)


def make_provider(
    batch_size: int = 1, embedding_size: int = EMBEDDING_DIMENSION
) -> tuple[GeminiEmbeddingProvider, MagicMock]:
    """Create a provider with a mocked client for testing."""
    mock_client = MagicMock()
    provider = GeminiEmbeddingProvider(
        client=mock_client,
        model=DEFAULT_EMBEDDING_MODEL,
        batch_size=batch_size,
    )
    return provider, mock_client


def test_embedding_provider_processes_texts_sequentially():
    """Embedding requests should be processed according to batch_size."""
    provider, mock_client = make_provider(batch_size=1)

    # Track call count for generating unique embeddings
    call_count = [0]

    def mock_embed_content(model, contents, config):
        call_count[0] += 1
        embeddings = [[float(call_count[0])] * EMBEDDING_DIMENSION for _ in contents]
        return make_embedding_response(embeddings)

    mock_client.models.embed_content.side_effect = mock_embed_content

    embeddings = provider.embed_texts(["a", "b", "c"])

    # With batch_size=1, each text is processed separately
    assert mock_client.models.embed_content.call_count == 3
    assert len(embeddings) == 3
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in embeddings)


def test_embedding_provider_processes_batch():
    """Embedding provider should process multiple texts in a batch."""
    provider, mock_client = make_provider(batch_size=3)

    mock_client.models.embed_content.return_value = make_embedding_response([
        [0.0] * EMBEDDING_DIMENSION,
        [1.0] * EMBEDDING_DIMENSION,
        [2.0] * EMBEDDING_DIMENSION,
    ])

    embeddings = provider.embed_texts(["a", "b", "c"])

    # With batch_size=3, all texts processed in one call
    assert mock_client.models.embed_content.call_count == 1
    assert len(embeddings) == 3


def test_embedding_provider_rejects_wrong_embedding_dimension():
    """Provider should reject embeddings with wrong dimensions."""
    provider, mock_client = make_provider()

    mock_client.models.embed_content.return_value = make_embedding_response([
        [0.0] * 100
    ])

    with pytest.raises(ValueError, match="Expected 1536 dimensions"):
        provider.embed_texts(["a"])


def test_embedding_provider_rejects_mismatched_response_count():
    """Provider should reject if API returns wrong number of embeddings."""
    provider, mock_client = make_provider(batch_size=2)

    # Return only one embedding when two were expected
    mock_client.models.embed_content.return_value = make_embedding_response([
        [0.0] * EMBEDDING_DIMENSION
    ])

    with pytest.raises(ValueError, match="response length"):
        provider.embed_texts(["a", "b"])


def test_embedding_provider_surfaces_external_failures():
    """Provider should surface API failures."""
    provider, mock_client = make_provider()

    mock_client.models.embed_content.side_effect = RuntimeError("embedding provider unavailable")

    with pytest.raises(RuntimeError, match="embedding provider unavailable"):
        provider.embed_texts(["a"])


def test_embedding_provider_from_env_uses_configuration(monkeypatch):
    """Provider should read configuration from environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "5")

    with patch("support_agent.embeddings.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        provider = GeminiEmbeddingProvider.from_env()

    mock_client_class.assert_called_once_with(api_key="test-key")
    assert provider._model == "models/gemini-embedding-001"
    assert provider._batch_size == 5


def test_embedding_provider_from_env_requires_api_key(monkeypatch):
    """Provider should require GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiEmbeddingProvider.from_env()


def test_embedding_provider_returns_empty_for_empty_input():
    """Provider should return empty list for empty input."""
    provider, mock_client = make_provider()

    assert provider.embed_texts([]) == []
    mock_client.models.embed_content.assert_not_called()


def test_embedding_provider_rejects_invalid_batch_size():
    """Provider should reject non-positive batch sizes."""
    mock_client = MagicMock()

    with pytest.raises(ValueError, match="batch_size must be positive"):
        GeminiEmbeddingProvider(client=mock_client, model="test-model", batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        GeminiEmbeddingProvider(client=mock_client, model="test-model", batch_size=-1)


def test_embedding_provider_uses_retrieval_document_task_type():
    """Provider should use RETRIEVAL_DOCUMENT task type for embeddings."""
    provider, mock_client = make_provider()

    mock_client.models.embed_content.return_value = make_embedding_response([
        [0.5] * EMBEDDING_DIMENSION
    ])

    provider.embed_texts(["a"])

    # Verify the task type was set correctly
    call_args = mock_client.models.embed_content.call_args
    assert call_args is not None
    config = call_args.kwargs.get("config")
    assert config is not None
    assert config.task_type == "RETRIEVAL_DOCUMENT"
