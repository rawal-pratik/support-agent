"""Tests for corpus ingestion."""

from pathlib import Path

import pytest

from support_agent.corpus import DocumentChunk
from support_agent.ingestion import ingest_corpus


class FakeEmbeddingProvider:
    """Fake embedding provider for testing."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.last_texts: list[str] | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.last_texts = list(texts)
        if self.fail:
            raise RuntimeError("embedding failed")
        return [[float(index)] * 1536 for index, _ in enumerate(texts)]


class FakeRepository:
    """Fake document chunk repository for testing."""

    def __init__(self, fail_on_insert: bool = False):
        self.fail_on_insert = fail_on_insert
        self.cleared = False
        self.inserted_chunks: list[DocumentChunk] | None = None
        self.inserted_embeddings: list[list[float]] | None = None

    def clear_chunks(self) -> None:
        self.cleared = True

    def insert_chunks_with_embeddings(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> list[dict]:
        if self.fail_on_insert:
            raise RuntimeError("database failed")
        self.inserted_chunks = list(chunks)
        self.inserted_embeddings = list(embeddings)
        return [{"id": chunk.chunk_id} for chunk in chunks]


def write_article(root: Path, relative: str, content: str) -> None:
    """Write a test article to the corpus."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_successful_corpus_ingestion_stores_chunks_and_embeddings(tmp_path: Path):
    """Successful ingestion should store chunks with their embeddings."""
    write_article(tmp_path, "a/doc-1.md", "one two three four five six")
    (tmp_path / "index.md").write_text(
        "# KB\n\n## A\n- [Doc 1](a/doc-1.md)\n",
        encoding="utf-8",
    )

    provider = FakeEmbeddingProvider()
    repository = FakeRepository()

    result = ingest_corpus(
        tmp_path / "index.md",
        repository,
        provider,
        chunk_size=3,
        overlap=1,
        clear_existing=True,
    )

    assert repository.cleared is True
    assert result.document_count == 1
    assert result.chunk_count == 3
    assert result.inserted_count == 3
    assert provider.last_texts == [
        "one two three",
        "three four five",
        "five six",
    ]
    assert [chunk.chunk_id for chunk in repository.inserted_chunks or []] == [
        "a/doc-1.md#chunk-0",
        "a/doc-1.md#chunk-1",
        "a/doc-1.md#chunk-2",
    ]
    assert len(repository.inserted_embeddings or []) == 3
    assert all(
        len(vector) == 1536 for vector in (repository.inserted_embeddings or [])
    )


def test_chunk_embedding_pairing_is_preserved(tmp_path: Path):
    """Each chunk should be paired with its corresponding embedding."""
    write_article(tmp_path, "b/doc-2.md", "alpha beta gamma delta")
    (tmp_path / "index.md").write_text(
        "# KB\n\n## B\n- [Doc 2](b/doc-2.md)\n",
        encoding="utf-8",
    )
    provider = FakeEmbeddingProvider()
    repository = FakeRepository()

    ingest_corpus(
        tmp_path / "index.md", repository, provider, chunk_size=2, overlap=0
    )

    assert [chunk.text for chunk in repository.inserted_chunks or []] == [
        "alpha beta",
        "gamma delta",
    ]
    assert repository.inserted_embeddings == [[0.0] * 1536, [1.0] * 1536]


def test_ingestion_surfaces_embedding_provider_failures(tmp_path: Path):
    """Embedding provider failures should be surfaced."""
    write_article(tmp_path, "doc.md", "text one two")
    (tmp_path / "index.md").write_text(
        "# KB\n\n## C\n- [Doc](doc.md)\n", encoding="utf-8"
    )
    provider = FakeEmbeddingProvider(fail=True)
    repository = FakeRepository()

    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest_corpus(tmp_path / "index.md", repository, provider)


def test_ingestion_surfaces_database_failures(tmp_path: Path):
    """Database failures should be surfaced."""
    write_article(tmp_path, "doc.md", "text one two")
    (tmp_path / "index.md").write_text(
        "# KB\n\n## C\n- [Doc](doc.md)\n", encoding="utf-8"
    )
    provider = FakeEmbeddingProvider()
    repository = FakeRepository(fail_on_insert=True)

    with pytest.raises(RuntimeError, match="database failed"):
        ingest_corpus(tmp_path / "index.md", repository, provider)


def test_empty_corpus_returns_zero_chunks_and_skips_embedding(tmp_path: Path):
    """Empty corpus should skip embedding and return zero counts."""
    (tmp_path / "index.md").write_text("# KB\n\n## Empty\n", encoding="utf-8")
    provider = FakeEmbeddingProvider()
    repository = FakeRepository()

    result = ingest_corpus(tmp_path / "index.md", repository, provider)

    assert repository.cleared is True
    assert result.document_count == 0
    assert result.chunk_count == 0
    assert result.inserted_count == 0
    assert provider.last_texts is None
    assert repository.inserted_chunks is None


def test_ingestion_can_skip_clearing_existing_chunks(tmp_path: Path):
    """Ingestion can skip clearing existing chunks when clear_existing=False."""
    write_article(tmp_path, "doc.md", "one two three")
    (tmp_path / "index.md").write_text(
        "# KB\n\n## D\n- [Doc](doc.md)\n", encoding="utf-8"
    )
    provider = FakeEmbeddingProvider()
    repository = FakeRepository()

    result = ingest_corpus(
        tmp_path / "index.md",
        repository,
        provider,
        clear_existing=False,
    )

    assert repository.cleared is False
    assert result.chunk_count == 1
    assert result.inserted_count == 1
