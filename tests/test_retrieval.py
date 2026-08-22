"""Tests for semantic retrieval."""

from unittest.mock import MagicMock

import pytest

from support_agent.embeddings import EMBEDDING_DIMENSION
from support_agent.models import Ticket
from support_agent.retrieval import RetrievedChunk, SemanticRetriever


def make_embedding_response(embedding: list[float]) -> list[list[float]]:
    """Create a single embedding response."""
    return [embedding]


def make_search_result(
    chunk_id: str,
    source_path: str,
    title: str,
    category: str,
    text: str,
    similarity: float,
) -> dict:
    """Create a fake search result from the database."""
    return {
        "id": chunk_id,
        "source_path": source_path,
        "title": title,
        "category": category,
        "chunk_text": text,
        "similarity": similarity,
    }


def make_retriever(
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    embeddings: list[list[float]] | None = None,
    search_results: list[dict] | None = None,
) -> tuple[SemanticRetriever, MagicMock, MagicMock]:
    """Create a retriever with mocked dependencies."""
    mock_embedding_provider = MagicMock()
    mock_repository = MagicMock()

    # Configure mocks
    if embeddings is not None:
        mock_embedding_provider.embed_texts.return_value = embeddings
    else:
        mock_embedding_provider.embed_texts.return_value = [[0.0] * EMBEDDING_DIMENSION]

    if search_results is not None:
        mock_repository.similarity_search.return_value = search_results
    else:
        mock_repository.similarity_search.return_value = []

    retriever = SemanticRetriever(
        embedding_provider=mock_embedding_provider,
        repository=mock_repository,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    return retriever, mock_embedding_provider, mock_repository


def test_retriever_embeds_ticket_text():
    """Retriever should embed the combined ticket text."""
    retriever, mock_embedding_provider, mock_repository = make_retriever()

    ticket = Ticket(subject="Login issue", issue="Cannot log in to my account")
    retriever.retrieve(ticket)

    # Verify embedding was called with combined text
    mock_embedding_provider.embed_texts.assert_called_once()
    call_args = mock_embedding_provider.embed_texts.call_args[0][0]
    assert len(call_args) == 1
    assert "Login issue" in call_args[0]
    assert "Cannot log in to my account" in call_args[0]


def test_retriever_calls_similarity_search_with_embedding():
    """Retriever should call similarity search with the query embedding."""
    test_embedding = [0.5] * EMBEDDING_DIMENSION
    retriever, mock_embedding_provider, mock_repository = make_retriever(
        embeddings=[test_embedding], top_k=10
    )

    ticket = Ticket(subject="Test", issue="Test issue")
    retriever.retrieve(ticket)

    mock_repository.similarity_search.assert_called_once()
    call_args = mock_repository.similarity_search.call_args
    assert call_args[0][0] == test_embedding
    assert call_args[1] == {"match_count": 10}


def test_retriever_returns_retrieved_chunks_with_metadata():
    """Retriever should return RetrievedChunk objects with full metadata."""
    search_results = [
        make_search_result(
            chunk_id="docs/auth.md#chunk-0",
            source_path="docs/auth.md",
            title="Authentication Guide",
            category="Security",
            text="To reset your password, click the forgot password link.",
            similarity=0.85,
        )
    ]

    retriever, _, _ = make_retriever(search_results=search_results)

    ticket = Ticket(subject="Password reset", issue="How do I reset my password?")
    chunks = retriever.retrieve(ticket)

    assert len(chunks) == 1
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].chunk_id == "docs/auth.md#chunk-0"
    assert chunks[0].source_path == "docs/auth.md"
    assert chunks[0].title == "Authentication Guide"
    assert chunks[0].category == "Security"
    assert chunks[0].text == "To reset your password, click the forgot password link."
    assert chunks[0].similarity == 0.85


def test_retriever_respects_top_k():
    """Retriever should request top_k results from the database."""
    retriever, _, mock_repository = make_retriever(top_k=3)

    ticket = Ticket(subject="Test", issue="Test issue")
    retriever.retrieve(ticket)

    call_args = mock_repository.similarity_search.call_args
    assert call_args[1]["match_count"] == 3


def test_retriever_filters_by_similarity_threshold():
    """Retriever should filter results below the similarity threshold."""
    search_results = [
        make_search_result(
            chunk_id="doc1.md",
            source_path="doc1.md",
            title="Doc 1",
            category="General",
            text="Highly relevant content",
            similarity=0.9,
        ),
        make_search_result(
            chunk_id="doc2.md",
            source_path="doc2.md",
            title="Doc 2",
            category="General",
            text="Moderately relevant content",
            similarity=0.6,
        ),
        make_search_result(
            chunk_id="doc3.md",
            source_path="doc3.md",
            title="Doc 3",
            category="General",
            text="Low relevance content",
            similarity=0.3,
        ),
    ]

    retriever, _, _ = make_retriever(
        search_results=search_results, similarity_threshold=0.5
    )

    ticket = Ticket(subject="Test", issue="Test issue")
    chunks = retriever.retrieve(ticket)

    # Only results with similarity >= 0.5 should be returned
    assert len(chunks) == 2
    assert chunks[0].similarity == 0.9
    assert chunks[1].similarity == 0.6


def test_retriever_returns_empty_when_no_results_meet_threshold():
    """Retriever should return empty list when no results meet threshold."""
    search_results = [
        make_search_result(
            chunk_id="doc.md",
            source_path="doc.md",
            title="Doc",
            category="General",
            text="Low relevance",
            similarity=0.2,
        )
    ]

    retriever, _, _ = make_retriever(
        search_results=search_results, similarity_threshold=0.8
    )

    ticket = Ticket(subject="Test", issue="Test issue")
    chunks = retriever.retrieve(ticket)

    assert chunks == []


def test_retriever_returns_empty_when_database_returns_empty():
    """Retriever should return empty list when database has no results."""
    retriever, _, _ = make_retriever(search_results=[])

    ticket = Ticket(subject="Test", issue="Test issue")
    chunks = retriever.retrieve(ticket)

    assert chunks == []


def test_retriever_returns_empty_when_embedding_provider_returns_empty():
    """Retriever should return empty list when embedding fails."""
    retriever, mock_embedding_provider, _ = make_retriever()
    mock_embedding_provider.embed_texts.return_value = []

    ticket = Ticket(subject="Test", issue="Test issue")
    chunks = retriever.retrieve(ticket)

    assert chunks == []


def test_retriever_surfaces_embedding_provider_failures():
    """Retriever should surface embedding provider errors."""
    retriever, mock_embedding_provider, _ = make_retriever()
    mock_embedding_provider.embed_texts.side_effect = RuntimeError(
        "embedding provider unavailable"
    )

    ticket = Ticket(subject="Test", issue="Test issue")

    with pytest.raises(RuntimeError, match="embedding provider unavailable"):
        retriever.retrieve(ticket)


def test_retriever_surfaces_database_failures():
    """Retriever should surface database errors."""
    retriever, _, mock_repository = make_retriever()
    mock_repository.similarity_search.side_effect = RuntimeError("database unavailable")

    ticket = Ticket(subject="Test", issue="Test issue")

    with pytest.raises(RuntimeError, match="database unavailable"):
        retriever.retrieve(ticket)


def test_retriever_preserves_order_by_similarity():
    """Retriever should preserve the similarity order from the database."""
    search_results = [
        make_search_result(
            chunk_id="doc1.md",
            source_path="doc1.md",
            title="Doc 1",
            category="General",
            text="First result",
            similarity=0.95,
        ),
        make_search_result(
            chunk_id="doc2.md",
            source_path="doc2.md",
            title="Doc 2",
            category="General",
            text="Second result",
            similarity=0.85,
        ),
        make_search_result(
            chunk_id="doc3.md",
            source_path="doc3.md",
            title="Doc 3",
            category="General",
            text="Third result",
            similarity=0.75,
        ),
    ]

    retriever, _, _ = make_retriever(search_results=search_results)

    ticket = Ticket(subject="Test", issue="Test issue")
    chunks = retriever.retrieve(ticket)

    # Order should be preserved (highest similarity first)
    assert len(chunks) == 3
    assert chunks[0].similarity == 0.95
    assert chunks[1].similarity == 0.85
    assert chunks[2].similarity == 0.75


def test_retriever_handles_ticket_with_empty_subject():
    """Retriever should handle tickets with empty subject."""
    retriever, mock_embedding_provider, _ = make_retriever()

    ticket = Ticket(subject="", issue="Cannot access my account")
    retriever.retrieve(ticket)

    # Verify embedding was called with just the issue text
    call_args = mock_embedding_provider.embed_texts.call_args[0][0]
    assert "Cannot access my account" in call_args[0]


def test_retriever_validates_top_k():
    """Retriever should validate top_k is positive."""
    mock_embedding_provider = MagicMock()
    mock_repository = MagicMock()

    with pytest.raises(ValueError, match="top_k must be positive"):
        SemanticRetriever(
            embedding_provider=mock_embedding_provider,
            repository=mock_repository,
            top_k=0,
        )

    with pytest.raises(ValueError, match="top_k must be positive"):
        SemanticRetriever(
            embedding_provider=mock_embedding_provider,
            repository=mock_repository,
            top_k=-1,
        )


def test_retriever_validates_similarity_threshold():
    """Retriever should validate similarity_threshold is between 0 and 1."""
    mock_embedding_provider = MagicMock()
    mock_repository = MagicMock()

    with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
        SemanticRetriever(
            embedding_provider=mock_embedding_provider,
            repository=mock_repository,
            similarity_threshold=-0.1,
        )

    with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
        SemanticRetriever(
            embedding_provider=mock_embedding_provider,
            repository=mock_repository,
            similarity_threshold=1.5,
        )


def test_retriever_handles_missing_similarity_in_result():
    """Retriever should handle results with missing similarity field."""
    search_results = [
        {
            "id": "doc.md",
            "source_path": "doc.md",
            "title": "Doc",
            "category": "General",
            "chunk_text": "Content",
            # similarity field is missing
        }
    ]

    retriever, _, _ = make_retriever(
        search_results=search_results, similarity_threshold=0.0
    )

    ticket = Ticket(subject="Test", issue="Test issue")
    chunks = retriever.retrieve(ticket)

    # Missing similarity defaults to 0.0, which passes threshold of 0.0
    assert len(chunks) == 1
    assert chunks[0].similarity == 0.0
