"""Semantic retrieval for support tickets."""

from typing import Protocol

from pydantic import BaseModel

from .db import DocumentChunkRepository
from .embeddings import EmbeddingProvider
from .models import Ticket


class RetrievedChunk(BaseModel):
    """A document chunk retrieved for a ticket with similarity score."""

    chunk_id: str
    source_path: str
    title: str
    category: str
    text: str
    similarity: float


class Retriever(Protocol):
    """Protocol for ticket retrieval."""

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        """Return relevant chunks for a ticket."""
        ...


class SemanticRetriever:
    """Semantic vector retrieval using embeddings and similarity search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: DocumentChunkRepository,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self._embedding_provider = embedding_provider
        self._repository = repository
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks for a ticket.

        Args:
            ticket: The support ticket to find relevant documentation for

        Returns:
            List of retrieved chunks with similarity >= threshold,
            ordered by similarity (highest first)
        """
        # Combine subject and issue for embedding
        ticket_text = f"{ticket.subject} {ticket.issue}".strip()

        # Embed the ticket as a retrieval query.
        query_embedding = self._embedding_provider.embed_query(ticket_text)
        if not query_embedding:
            return []

        # Search for similar chunks
        results = self._repository.similarity_search(
            query_embedding, match_count=self._top_k
        )

        # Filter by threshold and convert to RetrievedChunk objects
        retrieved_chunks: list[RetrievedChunk] = []
        for result in results:
            similarity = result.get("similarity", 0.0)
            if similarity >= self._similarity_threshold:
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=result.get("id", ""),
                        source_path=result.get("source_path", ""),
                        title=result.get("title", ""),
                        category=result.get("category", ""),
                        text=result.get("chunk_text", ""),
                        similarity=similarity,
                    )
                )

        return retrieved_chunks