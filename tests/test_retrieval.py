"""Basic functional tests for hybrid document retrieval."""

from unittest.mock import MagicMock

import pytest

from support_agent.models import Ticket
from support_agent.retrieval import (
    RetrievedChunk,
    SemanticRetriever,
)


def make_ticket(
    issue="How long do tests stay active in the system?",
    subject="Test expiration",
):
    return Ticket(
        subject=subject,
        issue=issue,
    )


def make_row(
    chunk_id,
    source_path,
    title,
    text,
    similarity=0.9,
    category="Screen / Managing Tests",
):
    return {
        "id": chunk_id,
        "source_path": source_path,
        "title": title,
        "category": category,
        "chunk_text": text,
        "similarity": similarity,
    }


def make_retriever(
    rows=None,
    top_k=2,
    candidate_k=5,
    similarity_threshold=0.0,
):
    embedding_provider = MagicMock()
    embedding_provider.embed_query.return_value = [0.1, 0.2]

    repository = MagicMock()
    repository.similarity_search.return_value = rows or []

    retriever = SemanticRetriever(
        embedding_provider=embedding_provider,
        repository=repository,
        top_k=top_k,
        candidate_k=candidate_k,
        similarity_threshold=similarity_threshold,
    )

    return retriever, embedding_provider, repository


class TestRetrieverConfiguration:
    def test_rejects_invalid_top_k(self):
        with pytest.raises(ValueError):
            SemanticRetriever(
                MagicMock(),
                MagicMock(),
                top_k=0,
            )

    def test_rejects_candidate_k_smaller_than_top_k(self):
        with pytest.raises(ValueError):
            SemanticRetriever(
                MagicMock(),
                MagicMock(),
                top_k=5,
                candidate_k=2,
            )

    def test_rejects_invalid_similarity_threshold(self):
        with pytest.raises(ValueError):
            SemanticRetriever(
                MagicMock(),
                MagicMock(),
                similarity_threshold=1.5,
            )


class TestRetrieve:
    def test_retrieves_matching_documents(self):
        rows = [
            make_row(
                "chunk-1",
                "tests/create-test.md",
                "Create a Test",
                "Create a new test from the Tests page.",
            ),
            make_row(
                "chunk-2",
                "tests/test-expiration.md",
                "Test Expiration",
                "Configure when a test expires.",
            ),
        ]

        retriever, embeddings, repository = make_retriever(
            rows=rows,
            top_k=2,
        )

        results = retriever.retrieve(
            make_ticket(
                issue="How do I configure test expiration?"
            )
        )

        assert len(results) == 2
        assert all(
            isinstance(result, RetrievedChunk)
            for result in results
        )

        embeddings.embed_query.assert_called_once()
        repository.similarity_search.assert_called_once()

    def test_returns_empty_when_embedding_is_empty(self):
        retriever, embeddings, repository = make_retriever()

        embeddings.embed_query.return_value = []

        result = retriever.retrieve(
            make_ticket()
        )

        assert result == []
        repository.similarity_search.assert_not_called()

    def test_applies_similarity_threshold(self):
        rows = [
            make_row(
                "high",
                "high.md",
                "Relevant Test",
                "Relevant documentation.",
                similarity=0.9,
            ),
            make_row(
                "low",
                "low.md",
                "Other Test",
                "Other documentation.",
                similarity=0.2,
            ),
        ]

        retriever, _, _ = make_retriever(
            rows=rows,
            top_k=5,
            similarity_threshold=0.5,
        )

        results = retriever.retrieve(
            make_ticket()
        )

        assert len(results) == 1
        assert results[0].chunk_id == "high"

    def test_respects_top_k(self):
        rows = [
            make_row(
                "chunk-1",
                "one.md",
                "Test One",
                "Test documentation.",
            ),
            make_row(
                "chunk-2",
                "two.md",
                "Test Two",
                "Test documentation.",
            ),
            make_row(
                "chunk-3",
                "three.md",
                "Test Three",
                "Test documentation.",
            ),
        ]

        retriever, _, _ = make_retriever(
            rows=rows,
            top_k=2,
        )

        results = retriever.retrieve(
            make_ticket()
        )

        assert len(results) == 2

    def test_deduplicates_chunks_from_same_source(self):
        rows = [
            make_row(
                "chunk-1",
                "same-article.md",
                "Test Expiration",
                "Configure test expiration.",
                similarity=0.9,
            ),
            make_row(
                "chunk-2",
                "same-article.md",
                "Test Expiration",
                "More information about expiration.",
                similarity=0.8,
            ),
            make_row(
                "chunk-3",
                "other-article.md",
                "Invite Candidates",
                "Invite candidates to tests.",
                similarity=0.7,
            ),
        ]

        retriever, _, _ = make_retriever(
            rows=rows,
            top_k=5,
        )

        results = retriever.retrieve(
            make_ticket()
        )

        sources = [
            result.source_path
            for result in results
        ]

        assert len(sources) == len(set(sources))
        assert "same-article.md" in sources
        assert "other-article.md" in sources

    def test_candidates_are_requested_from_repository(self):
        rows = [
            make_row(
                "chunk-1",
                "test.md",
                "Test",
                "Test documentation.",
            )
        ]

        retriever, _, repository = make_retriever(
            rows=rows,
            candidate_k=10,
        )

        retriever.retrieve(
            make_ticket()
        )

        repository.similarity_search.assert_called_once_with(
            [0.1, 0.2],
            match_count=10,
        )


class TestScoring:
    def test_title_relevance_can_affect_ranking(self):
        rows = [
            make_row(
                "generic",
                "generic.md",
                "General Settings",
                "Configure settings for your tests.",
                similarity=0.9,
            ),
            make_row(
                "specific",
                "expiration.md",
                "Test Expiration",
                "Configure when tests expire.",
                similarity=0.9,
            ),
        ]

        retriever, _, _ = make_retriever(
            rows=rows,
            top_k=1,
        )

        results = retriever.retrieve(
            make_ticket(
                issue="How do I configure test expiration?"
            )
        )

        assert len(results) == 1
        assert results[0].chunk_id == "specific"


class TestHelpers:
    def test_query_combines_subject_and_issue(self):
        ticket = make_ticket(
            subject="Test expiration",
            issue="How long does it stay active?",
        )

        query = SemanticRetriever._query(ticket)

        assert query == (
            "Test expiration. How long does it stay active?"
        )

    def test_query_uses_issue_when_subject_missing(self):
        ticket = Ticket(
            subject="",
            issue="How do I create a test?",
        )

        assert (
            SemanticRetriever._query(ticket)
            == "How do I create a test?"
        )

    def test_invalid_similarity_value_becomes_zero(self):
        assert SemanticRetriever._float(None) == 0.0
        assert SemanticRetriever._float("invalid") == 0.0
        assert SemanticRetriever._float("0.75") == 0.75