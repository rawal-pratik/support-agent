from unittest.mock import Mock, patch

import pytest

from support_agent.corpus import DocumentChunk
from support_agent.db import DocumentChunkRepository


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.inserted = None
        self.deleted_filter = None

    def insert(self, rows):
        self.inserted = rows
        return self

    def delete(self):
        return self

    def neq(self, column, value):
        self.deleted_filter = (column, value)
        return self

    def execute(self):
        return self.response


class FakeClient:
    def __init__(self):
        self.query = FakeQuery(FakeResponse([]))
        self.rpc_name = None
        self.rpc_params = None
        self.rpc_response = FakeResponse([])

    def table(self, name):
        self.table_name = name
        return self.query

    def rpc(self, name, params):
        self.rpc_name = name
        self.rpc_params = params
        return FakeQuery(self.rpc_response)


def make_chunk(chunk_id="article.md#chunk-0"):
    return DocumentChunk(
        chunk_id=chunk_id,
        source_path="article.md",
        title="Article",
        category="Category",
        text="Chunk text",
    )


def test_repository_from_env_creates_supabase_client(monkeypatch):
    with patch("support_agent.db.create_client", return_value=object()) as create_client:
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")

        repository = DocumentChunkRepository.from_env()

    assert isinstance(repository, DocumentChunkRepository)
    create_client.assert_called_once_with("https://example.supabase.co", "test-key")


def test_repository_requires_supabase_configuration(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_KEY"):
        DocumentChunkRepository.from_env()


def test_insert_chunks_maps_chunk_models_to_database_rows():
    client = FakeClient()
    client.query.response = FakeResponse([{"id": "article.md#chunk-0"}])
    repository = DocumentChunkRepository(client)

    inserted = repository.insert_chunks([make_chunk()])

    assert client.table_name == "document_chunks"
    assert client.query.inserted == [
        {
            "id": "article.md#chunk-0",
            "source_path": "article.md",
            "title": "Article",
            "category": "Category",
            "chunk_text": "Chunk text",
        }
    ]
    assert inserted == [{"id": "article.md#chunk-0"}]


def test_insert_chunks_does_not_call_database_for_empty_input():
    client = Mock()
    repository = DocumentChunkRepository(client)

    assert repository.insert_chunks([]) == []
    client.table.assert_not_called()


def test_clear_chunks_deletes_all_rows():
    client = FakeClient()
    repository = DocumentChunkRepository(client)

    repository.clear_chunks()

    assert client.table_name == "document_chunks"
    assert client.query.deleted_filter == ("id", "")


def test_similarity_search_sends_embedding_and_returns_rows():
    client = FakeClient()
    client.rpc_response = FakeResponse([{"id": "article.md#chunk-0", "similarity": 0.9}])
    repository = DocumentChunkRepository(client)

    matches = repository.similarity_search((0.1, 0.2), match_count=3)

    assert client.rpc_name == "match_document_chunks"
    assert client.rpc_params == {
        "query_embedding": [0.1, 0.2],
        "match_count": 3,
    }
    assert matches == [{"id": "article.md#chunk-0", "similarity": 0.9}]


def test_similarity_search_requires_positive_match_count():
    repository = DocumentChunkRepository(FakeClient())

    with pytest.raises(ValueError, match="match_count must be positive"):
        repository.similarity_search([0.1], match_count=0)


def test_repository_propagates_database_errors():
    client = FakeClient()
    client.query.insert = lambda rows: (_ for _ in ()).throw(RuntimeError("database down"))
    repository = DocumentChunkRepository(client)

    with pytest.raises(RuntimeError, match="database down"):
        repository.insert_chunks([make_chunk()])
