"""Basic tests for corpus ingestion."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.ingestion import ingest_corpus


def make_chunk(chunk_id: str, text: str = "chunk text"):
    chunk = MagicMock()
    chunk.chunk_id = chunk_id
    chunk.text = text
    return chunk


class TestIngestCorpus:
    @patch("support_agent.ingestion.chunk_documents")
    @patch("support_agent.ingestion.load_documents")
    def test_ingests_documents(
        self,
        mock_load_documents,
        mock_chunk_documents,
    ):
        documents = [MagicMock(), MagicMock()]

        chunks = [
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
        ]

        mock_load_documents.return_value = documents
        mock_chunk_documents.return_value = chunks

        repository = MagicMock()
        repository.get_existing_chunk_ids.return_value = set()
        repository.insert_chunks_with_embeddings.return_value = [
            {},
            {},
        ]

        embedding_provider = MagicMock()
        embedding_provider.embed_documents.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        result = ingest_corpus(
            "index.md",
            repository,
            embedding_provider,
        )

        assert result.document_count == 2
        assert result.chunk_count == 2
        assert result.inserted_count == 2
        assert result.skipped_count == 0

        repository.clear_chunks.assert_called_once()
        embedding_provider.embed_documents.assert_called_once_with(
            ["chunk text", "chunk text"]
        )
        repository.insert_chunks_with_embeddings.assert_called_once()


    @patch("support_agent.ingestion.chunk_documents")
    @patch("support_agent.ingestion.load_documents")
    def test_skips_existing_chunks(
        self,
        mock_load_documents,
        mock_chunk_documents,
    ):
        mock_load_documents.return_value = [MagicMock()]

        chunks = [
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
        ]
        mock_chunk_documents.return_value = chunks

        repository = MagicMock()
        repository.get_existing_chunk_ids.return_value = {"chunk-1"}
        repository.insert_chunks_with_embeddings.return_value = [{}]

        embedding_provider = MagicMock()
        embedding_provider.embed_documents.return_value = [
            [0.1, 0.2],
        ]

        result = ingest_corpus(
            "index.md",
            repository,
            embedding_provider,
        )

        assert result.chunk_count == 2
        assert result.inserted_count == 1
        assert result.skipped_count == 1

        embedding_provider.embed_documents.assert_called_once_with(
            ["chunk text"]
        )


    @patch("support_agent.ingestion.chunk_documents")
    @patch("support_agent.ingestion.load_documents")
    def test_does_not_clear_existing_when_disabled(
        self,
        mock_load_documents,
        mock_chunk_documents,
    ):
        mock_load_documents.return_value = [MagicMock()]
        mock_chunk_documents.return_value = [
            make_chunk("chunk-1")
        ]

        repository = MagicMock()
        repository.get_existing_chunk_ids.return_value = set()
        repository.insert_chunks_with_embeddings.return_value = [{}]

        embedding_provider = MagicMock()
        embedding_provider.embed_documents.return_value = [
            [0.1, 0.2]
        ]

        ingest_corpus(
            "index.md",
            repository,
            embedding_provider,
            clear_existing=False,
        )

        repository.clear_chunks.assert_not_called()


    @patch("support_agent.ingestion.chunk_documents")
    @patch("support_agent.ingestion.load_documents")
    def test_empty_corpus(
        self,
        mock_load_documents,
        mock_chunk_documents,
    ):
        mock_load_documents.return_value = []
        mock_chunk_documents.return_value = []

        repository = MagicMock()
        embedding_provider = MagicMock()

        result = ingest_corpus(
            "index.md",
            repository,
            embedding_provider,
        )

        assert result.document_count == 0
        assert result.chunk_count == 0
        assert result.inserted_count == 0
        assert result.skipped_count == 0

        embedding_provider.embed_documents.assert_not_called()
        repository.insert_chunks_with_embeddings.assert_not_called()


class TestIngestValidation:
    def test_invalid_batch_size(self):
        with pytest.raises(
            ValueError,
            match="batch_size must be positive",
        ):
            ingest_corpus(
                "index.md",
                MagicMock(),
                MagicMock(),
                batch_size=0,
            )

    def test_invalid_max_chunks(self):
        with pytest.raises(
            ValueError,
            match="max_chunks must be positive",
        ):
            ingest_corpus(
                "index.md",
                MagicMock(),
                MagicMock(),
                max_chunks=0,
            )


class TestIngestBatching:
    @patch("support_agent.ingestion.chunk_documents")
    @patch("support_agent.ingestion.load_documents")
    def test_processes_chunks_in_batches(
        self,
        mock_load_documents,
        mock_chunk_documents,
    ):
        mock_load_documents.return_value = [MagicMock()]

        mock_chunk_documents.return_value = [
            make_chunk("chunk-1", "one"),
            make_chunk("chunk-2", "two"),
            make_chunk("chunk-3", "three"),
        ]

        repository = MagicMock()
        repository.get_existing_chunk_ids.return_value = set()

        repository.insert_chunks_with_embeddings.side_effect = [
            [{}, {}],
            [{}],
        ]

        embedding_provider = MagicMock()
        embedding_provider.embed_documents.side_effect = [
            [[0.1], [0.2]],
            [[0.3]],
        ]

        result = ingest_corpus(
            "index.md",
            repository,
            embedding_provider,
            batch_size=2,
        )

        assert result.chunk_count == 3
        assert result.inserted_count == 3

        assert embedding_provider.embed_documents.call_count == 2
        assert (
            repository.insert_chunks_with_embeddings.call_count
            == 2
        )


class TestEmbeddingValidation:
    @patch("support_agent.ingestion.chunk_documents")
    @patch("support_agent.ingestion.load_documents")
    def test_rejects_wrong_embedding_count(
        self,
        mock_load_documents,
        mock_chunk_documents,
    ):
        mock_load_documents.return_value = [MagicMock()]

        mock_chunk_documents.return_value = [
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
        ]

        repository = MagicMock()
        repository.get_existing_chunk_ids.return_value = set()

        embedding_provider = MagicMock()
        embedding_provider.embed_documents.return_value = [
            [0.1],
        ]

        with pytest.raises(
            ValueError,
            match="does not match batch size",
        ):
            ingest_corpus(
                "index.md",
                repository,
                embedding_provider,
            )

        repository.insert_chunks_with_embeddings.assert_not_called()