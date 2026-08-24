"""Basic tests for Gemini embedding provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from support_agent.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    GeminiEmbeddingProvider,
)


def make_response(count: int = 1):
    return SimpleNamespace(
        embeddings=[
            SimpleNamespace(
                values=[0.1] * EMBEDDING_DIMENSION
            )
            for _ in range(count)
        ]
    )


def make_provider(client=None, **kwargs):
    return GeminiEmbeddingProvider(
        client or MagicMock(),
        **kwargs,
    )


class TestProviderConfiguration:
    def test_default_model(self):
        provider = make_provider()

        assert provider.model == DEFAULT_EMBEDDING_MODEL

    def test_invalid_batch_size(self):
        with pytest.raises(
            ValueError,
            match="batch_size",
        ):
            make_provider(batch_size=0)

    def test_invalid_tpm_configuration(self):
        with pytest.raises(
            ValueError,
            match="TPM",
        ):
            make_provider(
                tpm_limit=100,
                tpm_target=200,
            )

    def test_invalid_retry_configuration(self):
        with pytest.raises(
            ValueError,
            match="retry",
        ):
            make_provider(max_retries=-1)


class TestEnvironment:
    def test_from_env_requires_api_key(self, monkeypatch):
        monkeypatch.delenv(
            "GEMINI_API_KEY",
            raising=False,
        )

        with pytest.raises(
            ValueError,
            match="GEMINI_API_KEY must be configured",
        ):
            GeminiEmbeddingProvider.from_env()

    @patch("support_agent.embeddings.genai.Client")
    def test_from_env_creates_provider(
        self,
        mock_client,
        monkeypatch,
    ):
        monkeypatch.setenv(
            "GEMINI_API_KEY",
            "test-key",
        )

        provider = GeminiEmbeddingProvider.from_env()

        mock_client.assert_called_once_with(
            api_key="test-key"
        )
        assert isinstance(
            provider,
            GeminiEmbeddingProvider,
        )


class TestEmbedding:
    def test_embed_documents_empty_input(self):
        client = MagicMock()
        provider = make_provider(client)

        assert provider.embed_documents([]) == []

        client.models.count_tokens.assert_not_called()
        client.models.embed_content.assert_not_called()

    def test_embed_query_empty_input(self):
        client = MagicMock()
        provider = make_provider(client)

        assert provider.embed_query("   ") == []

        client.models.count_tokens.assert_not_called()
        client.models.embed_content.assert_not_called()

    def test_embed_documents(self):
        client = MagicMock()

        client.models.count_tokens.return_value = (
            SimpleNamespace(total_tokens=10)
        )

        client.models.embed_content.return_value = (
            make_response(2)
        )

        provider = make_provider(client)

        result = provider.embed_documents(
            [
                "first document",
                "second document",
            ]
        )

        assert len(result) == 2
        assert len(result[0]) == EMBEDDING_DIMENSION
        assert len(result[1]) == EMBEDDING_DIMENSION

        client.models.count_tokens.assert_called_once()
        client.models.embed_content.assert_called_once()

    def test_embed_query(self):
        client = MagicMock()

        client.models.count_tokens.return_value = (
            SimpleNamespace(total_tokens=10)
        )

        client.models.embed_content.return_value = (
            make_response(1)
        )

        provider = make_provider(client)

        result = provider.embed_query(
            "How do I create a test?"
        )

        assert len(result) == EMBEDDING_DIMENSION

        client.models.count_tokens.assert_called_once()
        client.models.embed_content.assert_called_once()


class TestBatching:
    def test_documents_are_processed_in_batches(self):
        client = MagicMock()

        client.models.count_tokens.side_effect = [
            SimpleNamespace(total_tokens=10),
            SimpleNamespace(total_tokens=10),
        ]

        client.models.embed_content.side_effect = [
            make_response(2),
            make_response(1),
        ]

        provider = make_provider(
            client,
            batch_size=2,
        )

        result = provider.embed_documents(
            [
                "one",
                "two",
                "three",
            ]
        )

        assert len(result) == 3
        assert client.models.count_tokens.call_count == 2
        assert client.models.embed_content.call_count == 2


class TestTokenLimits:
    def test_batch_over_tpm_limit_raises(self):
        client = MagicMock()

        client.models.count_tokens.return_value = (
            SimpleNamespace(total_tokens=101)
        )

        provider = make_provider(
            client,
            tpm_limit=100,
            tpm_target=100,
        )

        with pytest.raises(
            ValueError,
            match="exceeding TPM limit",
        ):
            provider.embed_documents(
                ["large document"]
            )

        client.models.embed_content.assert_not_called()


class TestRateLimitRetry:
    def test_429_error_is_retried(self):
        client = MagicMock()

        client.models.count_tokens.return_value = (
            SimpleNamespace(total_tokens=10)
        )

        rate_limit_error = Exception(
            "429 RESOURCE_EXHAUSTED"
        )

        client.models.embed_content.side_effect = [
            rate_limit_error,
            make_response(1),
        ]

        provider = make_provider(
            client,
            max_retries=1,
            base_backoff_seconds=0.01,
        )

        with patch(
            "support_agent.embeddings.time.sleep"
        ):
            result = provider.embed_query(
                "test query"
            )

        assert len(result) == EMBEDDING_DIMENSION
        assert (
            client.models.embed_content.call_count
            == 2
        )